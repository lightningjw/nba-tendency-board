"""The network behind `transformer.py`, and the only module in the package that imports PyTorch.

It runs in its OWN process. LightGBM and PyTorch each bring an OpenMP runtime, and on macOS two of
them in one process abort it, so the pipeline writes the per-game arrays to a file, starts
`python -m tendency.transformer_net fit ...`, and reads the probabilities back. This module must
therefore never import `models.py` (or anything else that loads LightGBM): the list of heads and
their class counts arrive in the payload instead.

    python -m tendency.transformer_net fit   <in.npz> <out.npz> <facts.json> [--max-epochs N] [--seed N] [--config NAME]
    python -m tendency.transformer_net check <in.npz>          # the leakage self-check, printed as JSON
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

SEED = 7
BATCH_GAMES = 16
MAX_EPOCHS = 40
PATIENCE = 4
LR = 2e-3
WEIGHT_DECAY = 0.01
MAX_THREADS = 8


@dataclass(frozen=True)
class Config:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    player_dim: int
    history: bool            # False: no attention to earlier positions and no previous-possession token (an ablation)


# The step allows three configurations and no more. The third is an ablation, not a bigger model:
# the same network with the sequence switched off, to see what the sequence itself is worth.
CONFIGS = [Config("small", 64, 2, 4, 0.1, 16, True),
           Config("wide", 128, 3, 4, 0.1, 16, True),
           Config("no_history", 64, 2, 4, 0.1, 16, False)]



@dataclass
class GameData:
    """Padded arrays, games x positions (x ...). Built by `transformer.build_game_data`."""
    xs: np.ndarray           # start-known features, standardised
    xn: np.ndarray           # features read at the deciding event, standardised
    xc: np.ndarray           # per-candidate features, games x positions x 5 x n
    pid: np.ndarray          # player index of each of the five (0 = not seen in training)
    y: dict                  # head -> label index, -1 where the head has no label
    logb: dict               # head -> log of the head's baseline probabilities
    mask: np.ndarray         # True where a real possession sits
    game_part: np.ndarray    # split of each game
    heads: list              # head names, in scorecard order
    n_classes: dict          # head -> number of classes

    def save(self, path) -> None:
        np.savez(path, xs=self.xs, xn=self.xn, xc=self.xc, pid=self.pid, mask=self.mask, game_part=self.game_part.astype(str),
                 meta=json.dumps({"heads": self.heads, "n_classes": self.n_classes}),
                 **{f"y_{h}": v for h, v in self.y.items()}, **{f"logb_{h}": v for h, v in self.logb.items()})

    @staticmethod
    def load(path) -> "GameData":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return GameData(z["xs"], z["xn"], z["xc"], z["pid"], {h: z[f"y_{h}"] for h in meta["heads"]}, {h: z[f"logb_{h}"] for h in meta["heads"]},
                        z["mask"], z["game_part"].astype(object), meta["heads"], meta["n_classes"])


# ---- the network -------------------------------------------------------------------------------
def build_model(cfg: Config, n_players: int, max_len: int, gd: "GameData", leaky: bool = False):
    import torch
    from torch import nn
    n_start, n_snap, n_cand = gd.xs.shape[-1], gd.xn.shape[-1], gd.xc.shape[-1]
    ALL_HEADS, N_CLASSES = list(gd.heads), dict(gd.n_classes)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.cfg = cfg
            self.players = nn.Embedding(n_players + 1, cfg.player_dim)
            self.start = nn.Linear(n_start, d)
            self.prev = nn.Linear(n_snap + cfg.player_dim, d)          # the possession before: its late context and its five
            self.prev_label = nn.ModuleDict({h: nn.Embedding(N_CLASSES[h] + 1, d) for h in ALL_HEADS})   # index 0 = no label
            self.prev_finisher = nn.Linear(cfg.player_dim, d, bias=False)       # who finished it
            self.bos = nn.Parameter(torch.zeros(d))
            self.pos = nn.Embedding(max_len, d)
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, dropout=cfg.dropout, batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(cfg.dropout)
            self.tempo = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_CLASSES["tempo"]))
            self.fuse = nn.Sequential(nn.Linear(d + n_snap + cfg.player_dim, d), nn.GELU())
            self.heads = nn.ModuleDict({h: nn.Linear(d, N_CLASSES[h]) for h in ("ending_type", "shot_family", "action_style")})
            self.finisher = nn.Sequential(nn.Linear(d + n_cand + cfg.player_dim, d), nn.GELU(), nn.Linear(d, 1))
            for last in (self.tempo[-1], self.finisher[-1], *self.heads.values()):
                nn.init.zeros_(last.weight)                                     # an untrained model is exactly the baseline
                nn.init.zeros_(last.bias)

        def forward(self, b: dict) -> dict:
            xs, xn, xc, pid, mask = b["xs"], b["xn"], b["xc"], b["pid"], b["mask"]
            B, L = mask.shape
            pl = self.drop(self.players(pid))                                   # B x L x 5 x p
            z = self.start(xs) + self.pos(torch.arange(L, device=xs.device))[None]
            if self.cfg.history:
                prev = self.prev(torch.cat([xn, pl.mean(dim=2)], dim=-1))
                for h in ALL_HEADS:
                    prev = prev + self.prev_label[h](b["y"][h] + 1)
                slot = b["y"]["finisher"].clamp(min=0)
                who = torch.gather(pl, 2, slot[..., None, None].expand(B, L, 1, pl.shape[-1])).squeeze(2)
                prev = prev + self.prev_finisher(who) * (b["y"]["finisher"] >= 0)[..., None]
                shifted = torch.cat([self.bos.expand(B, 1, -1), prev[:, :-1]], dim=1)      # position k receives possession k - 1
                if leaky:
                    shifted = prev                                              # the self-check's deliberate bug: k sees its own outcome
                z = z + shifted
                attn = torch.triu(torch.ones(L, L, dtype=torch.bool, device=xs.device), diagonal=1)     # True = may not look
                pad = ~mask                                                 # every row can still see position 0, which is real
            else:
                attn = ~torch.eye(L, dtype=torch.bool, device=xs.device)        # each position sees only itself
                pad = None                                                  # a padded row that may see nothing would turn into NaN
            h = self.norm(self.trunk(z, mask=attn, src_key_padding_mask=pad))
            out = {"tempo": b["logb"]["tempo"] + self.tempo(h)}
            f = self.fuse(torch.cat([h, xn, pl.mean(dim=2)], dim=-1))
            for name, lin in self.heads.items():
                out[name] = b["logb"][name] + lin(self.drop(f))
            per = torch.cat([f[:, :, None, :].expand(B, L, 5, f.shape[-1]), xc, pl], dim=-1)
            out["finisher"] = b["logb"]["finisher"] + self.finisher(per).squeeze(-1)
            return out

    return Net()


def _batch(gd: GameData, games: np.ndarray, trim: bool = True) -> dict:
    import torch
    t = torch.from_numpy
    keep = int(gd.mask[games].sum(axis=1).max()) if trim else gd.mask.shape[1]   # trim the padding this batch does not need
    return {"xs": t(gd.xs[games, :keep]), "xn": t(gd.xn[games, :keep]), "xc": t(gd.xc[games, :keep]), "pid": t(gd.pid[games, :keep]),
            "mask": t(gd.mask[games, :keep]), "y": {h: t(v[games, :keep]) for h, v in gd.y.items()},
            "logb": {h: t(v[games, :keep]) for h, v in gd.logb.items()}}


def _loss(out: dict, b: dict):
    import torch.nn.functional as F
    total = 0.0
    for h in out:
        y = b["y"][h]
        has = (y >= 0) & b["mask"]
        if has.any():
            total = total + F.cross_entropy(out[h][has], y[has])
    return total


def predict(model, gd: GameData, games: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Probabilities per head for every real possession of `games` (default: all), in Features.poss row order."""
    import torch
    games = np.arange(len(gd.mask)) if games is None else games
    model.eval()
    parts: dict[str, list] = {h: [] for h in gd.heads}
    with torch.no_grad():
        for i in range(0, len(games), 64):
            idx = games[i:i + 64]
            b = _batch(gd, idx)
            out = model(b)
            m = b["mask"].numpy()
            for h in gd.heads:
                parts[h].append(torch.softmax(out[h], dim=-1).numpy()[m])
    return {h: np.concatenate(v) for h, v in parts.items()}


def train(cfg: Config, gd: GameData, log=print, max_epochs: int | None = None, seed: int = SEED) -> tuple[object, dict]:
    """Fit one configuration on training games, stop on the validation games. Returns (model, facts about the fit)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_g, valid_g = np.flatnonzero(gd.game_part == "train"), np.flatnonzero(gd.game_part == "valid")
    model = build_model(cfg, int(gd.pid.max()), gd.mask.shape[1], gd)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def valid_loss() -> float:
        if not len(valid_g):
            return float("nan")
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(valid_g), 64):
                b = _batch(gd, valid_g[i:i + 64])
                tot += float(_loss(model(b), b)) * len(b["mask"])
                cnt += len(b["mask"])
        return tot / cnt

    t0 = time.time()
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs or MAX_EPOCHS):
        model.train()
        order = rng.permutation(train_g)
        for i in range(0, len(order), BATCH_GAMES):
            b = _batch(gd, np.sort(order[i:i + BATCH_GAMES]))
            opt.zero_grad()
            loss = _loss(model(b), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs = epoch + 1
        v = valid_loss()
        if not np.isfinite(v):                                                  # no validation games: keep the last epoch
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            continue
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    facts = {"name": cfg.name, **{k: v for k, v in asdict(cfg).items() if k != "name"}, "epochs": epochs,
             "valid_loss": None if not np.isfinite(best) else round(best, 5), "seconds": round(time.time() - t0, 1),
             "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    log(f"transformer: {cfg.name}: {facts['n_parameters']:,} parameters, {epochs} epochs, {facts['seconds']:.0f} s, "
        f"validation loss {facts['valid_loss']}")
    return model, facts



# ---- the leakage self-check ---------------------------------------------------------------------
def leak_check(gd: GameData, k: int = 40, leaky: bool = False) -> dict:
    """Perturb possession k onward in every game and report how far predictions BEFORE k move (they must not).

    Three perturbations, on a randomly initialised network in eval mode with its zeroed output layers
    re-randomised (an untrained model would return the baseline whatever it was fed, and pass trivially):

    * `labels_from_k`: rewrite every label of possessions >= k. Predictions for possessions <= k must hold:
      possession k's own outcome may not reach its own prediction.
    * `context_after_k`: rewrite all features of possessions > k. Predictions for possessions <= k must hold.
    * `late_context_at_k`: rewrite what is read at k's deciding event (and its five). The tempo prediction
      for k must hold; the other heads are allowed to use it and must move (reported as `teeth_late`).

    `untrained_minus_baseline` is measured first, on the fresh network: its logits must equal the log baseline.

    `teeth_earlier` rewrites labels BEFORE k and reports how far later predictions move: it must be > 0, or
    the model is not reading the sequence and the zeros above would mean nothing. `leaky=True` breaks the
    model on purpose (possession k's own labels are fed at position k) to show the check catches it.
    """
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    games = np.flatnonzero(gd.mask.sum(axis=1) > k + 5)[:8]
    model = build_model(CONFIGS[0], int(gd.pid.max()), gd.mask.shape[1], gd, leaky=leaky)
    fresh = _raw(model, gd, games)                              # before any training the output layers are zero: logits = log baseline
    untrained = float(max(np.abs(fresh[h] - gd.logb[h][games])[gd.mask[games]].max() for h in gd.heads))
    for p in model.parameters():
        if float(p.abs().sum()) == 0.0:
            torch.nn.init.normal_(p, std=0.2)
    base = {h: v for h, v in _raw(model, gd, games).items()}

    def moved(alt: GameData, upto: int, heads=None, after: bool = False) -> float:
        out = _raw(model, alt, games)
        sl = slice(upto + 1, None) if after else slice(0, upto + 1)
        return float(max(np.abs(out[h][:, sl] - base[h][:, sl]).max() for h in (heads or gd.heads)))

    def copy() -> GameData:
        return GameData(gd.xs.copy(), gd.xn.copy(), gd.xc.copy(), gd.pid.copy(), {h: v.copy() for h, v in gd.y.items()},
                        {h: v.copy() for h, v in gd.logb.items()}, gd.mask, gd.game_part, gd.heads, gd.n_classes)

    def scramble_labels(a: GameData, sl: slice) -> None:
        for h in a.heads:
            a.y[h][:, sl] = rng.integers(0, a.n_classes[h], size=a.y[h][:, sl].shape)

    labels = copy(); scramble_labels(labels, slice(k, None))
    later = copy()
    for arr in (later.xs, later.xn, later.xc):
        arr[:, k + 1:] = rng.standard_normal(arr[:, k + 1:].shape).astype(arr.dtype)
    later.pid[:, k + 1:] = rng.integers(0, int(gd.pid.max()) + 1, size=later.pid[:, k + 1:].shape)
    late = copy()
    late.xn[:, k] = rng.standard_normal(late.xn[:, k].shape).astype(late.xn.dtype)
    late.xc[:, k] = rng.standard_normal(late.xc[:, k].shape).astype(late.xc.dtype)
    late.pid[:, k] = rng.integers(0, int(gd.pid.max()) + 1, size=late.pid[:, k].shape)
    earlier = copy(); scramble_labels(earlier, slice(0, k))
    others = [h for h in gd.heads if h != "tempo"]
    return {"k": k, "n_games": int(len(games)), "untrained_minus_baseline": untrained,
            "labels_from_k": moved(labels, k), "context_after_k": moved(later, k),
            "late_context_at_k_tempo": moved(late, k, ["tempo"]), "teeth_late": moved(late, k, others),
            "teeth_earlier": moved(earlier, k - 1, after=True)}


def _raw(model, gd: GameData, games: np.ndarray) -> dict:
    """Logits per head for a few games, padded (games x positions x classes), with the baseline offsets left in."""
    import torch
    model.eval()
    with torch.no_grad():
        out = model(_batch(gd, games, trim=False))
    return {h: v.numpy() for h, v in out.items()}


def cpu_name() -> str:
    """The processor's marketing name where the system will say it, else the architecture."""
    import subprocess
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or platform.machine()
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def main(argv: list[str]) -> int:
    import torch
    mode, src = argv[0], argv[1]
    gd = GameData.load(src)
    if mode == "check":
        print(json.dumps({"clean": leak_check(gd), "broken_on_purpose": leak_check(gd, leaky=True)}))
        return 0
    out_npz, out_json = argv[2], argv[3]
    max_epochs = int(argv[argv.index("--max-epochs") + 1]) if "--max-epochs" in argv else None
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else SEED          # a second seed is a reproducibility check, not a fourth configuration
    threads = min(MAX_THREADS, torch.get_num_threads())
    torch.set_num_threads(threads)
    t0 = time.time()
    only = argv[argv.index("--config") + 1] if "--config" in argv else None          # one named configuration (used to score the ablation on its own)
    todo = [c for c in CONFIGS[:3] if only in (None, c.name)]                          # three and no more
    fits = [train(cfg, gd, lambda m: print(m, flush=True), max_epochs, seed) for cfg in todo]
    scored = [f for f in fits if f[1]["valid_loss"] is not None] or fits
    model, chosen = min(scored, key=lambda f: f[1]["valid_loss"] if f[1]["valid_loss"] is not None else 0.0)
    probs = predict(model, gd)
    # The kept configuration once more with another seed. Not a fourth configuration: the same one, to see whether
    # a verdict against the trees is the model or the dice. `transformer.attach` needs a win from both.
    kept = next(c for c in CONFIGS if c.name == chosen["name"])
    again, again_facts = train(kept, gd, lambda m: print(m.replace(kept.name + ":", kept.name + " (second seed):"), flush=True), max_epochs, seed + 1)
    probs.update({f"second_{h}": v for h, v in predict(again, gd).items()})
    facts = {"chosen": chosen["name"], "selection": "lowest loss on the validation games; held-out games were not looked at",
             "seeds": [seed, seed + 1], "second_seed_valid_loss": again_facts["valid_loss"],
             "hardware": f"{cpu_name()} CPU, {threads} threads, PyTorch {torch.__version__}",
             "seconds": round(time.time() - t0, 1), "n_train_games": int((gd.game_part == "train").sum()), "n_parameters": chosen["n_parameters"],
             "configs": [{"name": f["name"], "d_model": f["d_model"], "n_layers": f["n_layers"], "history": f["history"], "epochs": f["epochs"],
                          "n_parameters": f["n_parameters"], "seconds": f["seconds"], "valid_loss": f["valid_loss"], "chosen": f["name"] == chosen["name"]}
                         for _, f in fits]}
    np.savez(out_npz, **probs)
    with open(out_json, "w") as fh:
        json.dump(facts, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
