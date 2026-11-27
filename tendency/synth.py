"""Synthetic raw games in the ASSUMED cdn.nba.com shape.

Purpose: let the whole pipeline (possessions -> features -> models -> export) and
the app run without network access. Everything here is invented: placeholder
teams ("Sample Alders"), placeholder players ("Alder #4"), seasons 2090 and 2091.
Every file carries a top-level `_synthetic: true` marker and the exporter refuses
to label output "real" when it sees one.

A few situational quirks are planted on purpose (see QUIRKS) so the team report has
something true to find. Team index 7 has none and acts as a control.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from . import cdn_fields as F

TEAMS = [("ALD", "Alder", "Alders"), ("BIR", "Birch", "Birches"), ("CED", "Cedar", "Cedars"), ("DOG", "Dogwood", "Dogwoods"),
         ("ELM", "Elm", "Elms"), ("FIR", "Fir", "Firs"), ("GKO", "Ginkgo", "Ginkgos"), ("HZL", "Hazel", "Hazels")]
SEASONS = {2090: 4, 2091: 4}          # season start year -> times each pair meets
FAMS = ["rim", "paint", "midrange", "corner3", "above_break3"]
ENDS = ["shot", "foul_drawn", "turnover"]
MAKE_P = {"rim": 0.64, "paint": 0.44, "midrange": 0.41, "corner3": 0.39, "above_break3": 0.355}
ARCHETYPES = {
    "big": np.array([2.2, 1.6, 0.7, 0.15, 0.25]),
    "wing": np.array([0.9, 0.8, 0.9, 2.0, 1.2]),
    "guard": np.array([0.8, 0.9, 1.4, 0.7, 1.6]),
}
# team index -> (situation, effect). Planted so the tendency table has ground truth.
QUIRKS = {
    0: ("after_timeout", "corner3 x2.5"),
    1: ("clutch", "player 0 usage x2.2"),
    2: ("start_def_rebound", "rim x1.8"),
    3: ("after_timeout", "turnover x1.8"),
    4: ("q4", "foul_drawn x1.6"),
    5: ("start_turnover", "above_break3 x1.7"),
    6: ("def_in_penalty", "player 1 usage x1.8"),
}


# Step 2.2 labels need something to label. Shot descriptors by subType, with weights; "" is no descriptor.
DESCRIPTORS = {
    "three": (["", "pullup", "step back", "running"], [0.62, 0.22, 0.12, 0.04]),
    "mid": (["pullup", "", "fadeaway", "turnaround fadeaway", "step back", "turnaround"], [0.40, 0.15, 0.15, 0.10, 0.10, 0.10]),
    "float": (["driving floating", "floating"], [0.7, 0.3]),
    "Layup": (["driving", "running", "cutting", "", "driving finger roll"], [0.47, 0.18, 0.14, 0.09, 0.12]),
    "Dunk": (["cutting", "running", "driving", "alley-oop", ""], [0.30, 0.25, 0.20, 0.15, 0.10]),
    "Hook": (["turnaround", "", "driving"], [0.5, 0.3, 0.2]),
}
ARCH_STYLE = {"guard": {"pullup": 1.7, "step back": 1.6, "": 0.8}, "wing": {"": 1.4, "cutting": 1.3}, "big": {"cutting": 1.5, "pullup": 0.4, "step back": 0.3}}
ASSIST_P = {"cutting": 0.97, "alley-oop": 1.0, "": 0.85, "pullup": 0.3, "step back": 0.25, "driving": 0.45, "driving finger roll": 0.45,
            "driving floating": 0.5, "floating": 0.5, "running": 0.7, "putback": 0.0, "tip": 0.0}
STYLE_QUIRK = (2, "q4", "pullup", 3.0)       # team index 2 takes three times its usual share of pull-ups in the fourth quarter
QUICK_HOOK_TEAM = 5                          # step 2.4: this bench takes a player out after a foul more readily than the rest


def _team_id(i: int) -> int:
    return 9900001 + i


def _player_id(ti: int, pi: int) -> int:
    return 9910000 + ti * 100 + pi + 1


def make_league(rng: np.random.Generator) -> list[dict]:
    league = []
    for ti, (tri, one, name) in enumerate(TEAMS):
        end_mix = rng.dirichlet(np.array([0.745, 0.115, 0.14]) * 220)
        shot_mix = rng.dirichlet(np.array([0.30, 0.17, 0.11, 0.10, 0.32]) * 60)
        players = []
        for pi in range(10):
            arch = ["guard", "guard", "wing", "wing", "big", "guard", "wing", "big", "wing", "big"][pi]
            base_usage = [1.9, 1.45, 1.1, 0.85, 0.8, 1.0, 0.8, 0.7, 0.6, 0.55][pi]
            pm = shot_mix * ARCHETYPES[arch] * rng.lognormal(0, 0.2, 5)
            pe = end_mix * np.array([1.0, rng.lognormal(0, 0.3), rng.lognormal(0, 0.3)])
            players.append({
                "player_id": _player_id(ti, pi), "name": f"{one} #{pi * 3 + 1}", "arch": arch,
                "usage": base_usage * rng.lognormal(0, 0.12), "shot_mix": pm / pm.sum(), "end_mix": pe / pe.sum(),
                "skill": rng.normal(1.0, 0.05), "minutes_w": [3.4, 3.2, 3.0, 2.8, 2.6, 1.6, 1.4, 1.2, 0.9, 0.7][pi],
            })
        league.append({"idx": ti, "team_id": _team_id(ti), "tricode": tri, "city": "Sample", "name": name,
                       "end_mix": end_mix, "shot_mix": shot_mix, "players": players, "pace": 1.0})
    for t, pace in zip(league, rng.lognormal(0, 0.10, len(league))):     # drawn last, so the teams above are the ones the older tests know
        t["pace"] = float(pace)                              # some teams play faster; the tempo labels pick it up
    return league


def drift(league: list[dict], rng: np.random.Generator, team_sd: float = 0.0) -> list[dict]:
    """A later version of the league: same players, slightly different habits.

    Between seasons each player changes on his own. At mid-season (`team_sd` > 0) a whole team also
    shifts together, as after a trade or a new rotation, so the tendency layer's drift term has
    something real to measure in the sample.
    """
    out = []
    for t in league:
        t2 = {**t, "players": []}
        team_shot = rng.lognormal(0, team_sd, 5) if team_sd else np.ones(5)
        team_end = rng.lognormal(0, team_sd, 3) if team_sd else np.ones(3)
        for p in t["players"]:
            sm = p["shot_mix"] * rng.lognormal(0, 0.12, 5) * team_shot
            em = p["end_mix"] * team_end
            t2["players"].append({**p, "usage": p["usage"] * rng.lognormal(0, 0.15), "shot_mix": sm / sm.sum(), "end_mix": em / em.sum()})
        out.append(t2)
    return out


def _clock(s: float) -> str:
    s = max(0.0, round(s, 2))
    return f"PT{int(s // 60):02d}M{s - 60 * (s // 60):05.2f}S"


class GameSim:
    def __init__(self, game_id: str, tipoff: datetime, home: dict, away: dict, rng: np.random.Generator):
        self.gid, self.tip, self.rng = game_id, tipoff, rng
        self.T = {home["team_id"]: home, away["team_id"]: away}
        self.home, self.away = home["team_id"], away["team_id"]
        self.other = {self.home: self.away, self.away: self.home}
        self.actions: list[dict] = []
        self.score = {self.home: 0, self.away: 0}
        self.on = {tid: list(range(5)) for tid in self.T}      # indexes into players
        self.fouls = {tid: 0 for tid in self.T}
        self.tos = {tid: 0 for tid in self.T}
        self.period, self.clock, self.elapsed = 0, 720.0, 0.0
        self.pending_timeout = False
        # ---- step 2.4: what benches do at stoppages. A second generator, so the possession stream keeps its draws.
        self.rc = np.random.default_rng(int(game_id))
        self.pf: dict[tuple[int, int], int] = {}             # (team, player index) -> personal fouls
        self.out = {tid: set() for tid in self.T}            # fouled out
        self.sit: list[tuple[int, int]] = []                 # players to take out at the next dead ball (offensive fouls)
        self.period_tos = 0
        self.unanswered = {tid: 0 for tid in self.T}
        self._seen = dict(self.score)

    # -- event plumbing ---------------------------------------------------------
    def ev(self, atype: str, sub: str = "", team: int | None = None, pi: int | None = None, **extra) -> dict:
        n = len(self.actions) + 1
        a = {
            F.ACTION_NUMBER: n, F.ORDER_NUMBER: n * 10000, F.CLOCK: _clock(self.clock),
            "timeActual": (self.tip + timedelta(seconds=self.elapsed * 2.6)).strftime("%Y-%m-%dT%H:%M:%S.0Z"),
            F.PERIOD: self.period, "periodType": "REGULAR" if self.period <= 4 else "OVERTIME",
            F.ACTION_TYPE: atype, F.SUB_TYPE: sub, F.DESCRIPTOR: extra.pop("descriptor", ""),
            "qualifiers": extra.pop("qualifiers", []), F.PERSON_ID: 0,
            F.SCORE_HOME: str(self.score[self.home]), F.SCORE_AWAY: str(self.score[self.away]),
            F.DESCRIPTION: "",
        }
        if team is not None:
            a[F.TEAM_ID] = team
            a["teamTricode"] = self.T[team]["tricode"]
            if pi is not None:
                p = self.T[team]["players"][pi]
                a[F.PERSON_ID] = p["player_id"]
                a["playerName"] = p["name"]
        a.update(extra)
        a[F.DESCRIPTION] = extra.get("description_override") or f"{a.get('playerName', a.get('teamTricode', ''))} {atype} {sub}".strip()
        a.pop("description_override", None)
        personal = atype == "foul" and sub in ("personal", "offensive") and team is not None and pi is not None
        if personal:
            self.pf[(team, pi)] = self.pf.get((team, pi), 0) + 1
            a[F.DESCRIPTION] += f" FOUL ({self.pf[(team, pi)]} PF)"      # the real feed prints the running count the same way
        self.actions.append(a)
        if personal:
            self.after_foul(team, pi, defer=sub == "offensive")
        return a

    def after_foul(self, team: int, pi: int, defer: bool) -> None:
        """Foul trouble, as planted: a player with more fouls than the period number usually sits; the sixth is the last."""
        n = self.pf[(team, pi)]
        over = n - min(self.period, 4)
        p = 1.0 if n >= 6 else (0.05 if over < 0 else 0.12 if over == 0 else 0.55 if over == 1 else 0.85)
        if self.T[team]["idx"] == QUICK_HOOK_TEAM and n < 6:
            p = min(1.0, p + 0.3)
        if n >= 6:
            self.out[team].add(pi)
        if self.rc.random() < p:
            if defer:
                self.sit.append((team, pi))                  # the turnover that goes with the foul is logged first
            else:
                self.take_out(team, pi)

    def take_out(self, team: int, pi: int) -> None:
        if pi not in self.on[team]:
            return
        bench = [i for i in range(10) if i not in self.on[team] and i not in self.out[team]]
        if not bench:
            return
        w = np.array([self.T[team]["players"][i]["minutes_w"] for i in bench])
        new = int(self.rc.choice(bench, p=w / w.sum()))
        self.set_lineup(team, [i for i in self.on[team] if i != pi] + [new])

    def pid(self, team: int, pi: int) -> int:
        return self.T[team]["players"][pi]["player_id"]

    def tick(self, lo: float, hi: float) -> None:
        d = min(self.rng.uniform(lo, hi), max(0.0, self.clock - 0.1))
        self.clock -= d
        self.elapsed += d

    # -- substitutions / dead balls ----------------------------------------------
    def set_lineup(self, team: int, new: list[int]) -> None:
        cur = self.on[team]
        outs, ins = [p for p in cur if p not in new], [p for p in new if p not in cur]
        for o, i in zip(outs, ins):
            self.ev("substitution", "out", team, o)
            self.ev("substitution", "in", team, i)
        self.on[team] = sorted(new)

    def pick_lineup(self, team: int, closing: bool = False) -> list[int]:
        w = np.array([p["minutes_w"] for p in self.T[team]["players"]]) ** (2.5 if closing else 1.0)
        if self.out[team] and len(self.out[team]) <= 5:
            w[sorted(self.out[team])] = 0.0                  # fouled out
        return sorted(self.rng.choice(10, size=5, replace=False, p=w / w.sum()).tolist())

    def dead_ball(self, scored_on: int | None) -> None:
        r = self.rng
        late = self.period >= 4 and self.clock < 300
        for team in (self.home, self.away):                  # the run in progress: points since the other team last scored
            gain = self.score[team] - self._seen[team]
            if gain > 0:
                self.unanswered[team] += gain
                self.unanswered[self.other[team]] = 0
        self._seen = dict(self.score)
        for team, pi in self.sit:
            self.take_out(team, pi)
        self.sit = []
        due = self.period <= 4 and ((self.clock <= F.MANDATORY_FIRST_S and self.period_tos < 1)
                                    or (self.clock <= F.MANDATORY_SECOND_S and self.period_tos < 2))
        if due:                                              # the scorer charges it: home first, then whoever has one
            team = self.home if self.tos[self.home] <= self.tos[self.away] else self.away
            if self.tos[team] < F.TIMEOUTS_PER_GAME:
                self.tos[team] += 1
                self.period_tos += 1
                self.ev("timeout", "full", team, qualifiers=["team", "mandatory"])
                self.pending_timeout = True
        for team in (self.home, self.away):
            run = min(self.unanswered[self.other[team]], 10)  # planted: a bench calls time more often the longer the other team's run
            p_to = 0.02 + (0.05 if late else 0.0) + (0.02 if team == scored_on else 0.0) + 0.012 * run
            if self.tos[team] < F.TIMEOUTS_PER_GAME and r.random() < p_to:
                self.tos[team] += 1
                self.period_tos += 1
                self.ev("timeout", "full", team, qualifiers=["team"])
                self.pending_timeout = True
        if self.rc.random() < 0.004:                         # a challenge: the feed logs it as a timeout and nothing more
            self.period_tos += 1
            self.ev("timeout", "challenge", self.home if self.rc.random() < 0.5 else self.away, qualifiers=["team"])
            self.pending_timeout = True
        if r.random() < 0.006:                               # technical foul + technical free throw
            bad = self.home if r.random() < 0.5 else self.away
            good = self.other[bad]
            self.ev("foul", "technical", bad, int(r.choice(self.on[bad])))
            shooter = self.on[good][0]
            made = r.random() < 0.85
            if made:
                self.score[good] += 1
            self.ev("freethrow", "1 of 1", good, shooter, descriptor="technical", shotResult="Made" if made else "Missed",
                    description_override=f"{self.T[good]['players'][shooter]['name']} Free Throw Technical")
        for team in (self.home, self.away):
            if self.pending_timeout or r.random() < 0.10:
                self.set_lineup(team, self.pick_lineup(team, closing=late))

    # -- one possession -----------------------------------------------------------
    def situation(self, off: int, start_type: str) -> dict:
        margin = self.score[off] - self.score[self.other[off]]
        return {
            "after_timeout": self.pending_timeout,
            "clutch": self.period >= 4 and self.clock <= 300 and abs(margin) <= 5,
            "start_def_rebound": start_type == "def_rebound",
            "start_turnover": start_type == "turnover",
            "q4": self.period == 4,
            "def_in_penalty": self.fouls[self.other[off]] >= F.PENALTY_TEAM_FOULS,
        }

    def shot_geometry(self, fam: str) -> dict:
        r = self.rng
        if fam == "rim":
            d, ang = r.uniform(0, 3.5), r.uniform(0, np.pi)
            x, y, area = d * np.cos(ang) * 10, d * np.sin(ang) * 10, "Restricted Area"
        elif fam == "paint":
            x, y, area = r.uniform(-75, 75), r.uniform(45, 135), "In The Paint (Non-RA)"
        elif fam == "midrange":
            ang, d = r.uniform(0.15, np.pi - 0.15), r.uniform(15, 21)
            x, y, area = d * np.cos(ang) * 10, d * np.sin(ang) * 10, "Mid-Range"
        elif fam == "corner3":
            side = 1 if r.random() < 0.5 else -1
            x, y, area = side * r.uniform(222, 245), r.uniform(-30, 85), ("Right" if side > 0 else "Left") + " Corner 3"
        elif fam == "above_break3":
            ang, d = r.uniform(0.45, np.pi - 0.45), r.uniform(24.2, 28.5)
            x, y, area = d * np.cos(ang) * 10, d * np.sin(ang) * 10, "Above the Break 3"
        else:                                                # heave
            x, y, area = r.uniform(-200, 200), r.uniform(450, 700), "Backcourt"
        return {F.X_LEGACY: int(x), F.Y_LEGACY: int(y), F.AREA: area,
                F.SHOT_DISTANCE: round(float((x * x + y * y) ** 0.5) / 10, 2), "isFieldGoal": 1}

    def free_throws(self, off: int, shooter: int, n: int, tags: list[str] | None = None) -> bool:
        """Shoot n free throws. Returns True if the last one was made."""
        made = False
        for k in range(1, n + 1):
            made = self.rng.random() < 0.77
            if made:
                self.score[off] += 1
            self.ev("freethrow", f"{k} of {n}", off, shooter, shotResult="Made" if made else "Missed", qualifiers=list(tags or []))
        return made

    def descriptor(self, sub: str, fam: str, is3: bool, arch: str, second_chance: bool, q_style: bool) -> str:
        r = self.rng
        if second_chance and sub in ("Layup", "Dunk") and r.random() < 0.5:
            return "putback" if r.random() < 0.55 else "tip"
        key = "three" if is3 else "mid" if fam == "midrange" else "float" if sub == "Jump Shot" else sub
        words, w = DESCRIPTORS[key]
        w = np.array(w) * np.array([ARCH_STYLE[arch].get(x, 1.0) for x in words])
        if q_style:
            w = w * np.array([STYLE_QUIRK[3] if x == STYLE_QUIRK[2] else 1.0 for x in words])
        return str(r.choice(words, p=w / w.sum()))

    def rebound(self, off: int, p_off: float) -> bool:
        """Emit a rebound. Returns True if the offense kept the ball."""
        r = self.rng
        self.tick(0.5, 2.0)
        if r.random() < p_off:
            if r.random() < 0.1:
                self.ev("rebound", "offensive", off)        # team rebound, personId 0
            else:
                self.ev("rebound", "offensive", off, int(r.choice(self.on[off])))
            return True
        d = self.other[off]
        self.ev("rebound", "defensive", d, int(r.choice(self.on[d])))
        return False

    def possession(self, off: int, start_type: str) -> str:
        """Play one possession. Returns the next possession's start type."""
        r, T, d = self.rng, self.T[off], self.other[off]
        if self.clock < 4.0:                                 # end-of-period heave, or the clock just runs out
            if r.random() < 0.6:
                self.tick(0.2, max(0.3, self.clock - 0.4))
                shooter = int(r.choice(self.on[off]))
                made = r.random() < 0.03
                if made:
                    self.score[off] += 3
                self.ev("3pt", "Jump Shot", off, shooter, shotResult="Made" if made else "Missed", **self.shot_geometry("heave"))
            self.elapsed += self.clock
            self.clock = 0.0
            return "period_start"
        sit = self.situation(off, start_type)
        self.pending_timeout = False
        quirk = QUIRKS.get(T["idx"], (None, None))[0]
        q_on = bool(quirk and sit[quirk])
        quick = start_type in ("turnover", "def_rebound")
        dur = (r.uniform(2, 16) if quick else r.uniform(7, 23)) / T["pace"]
        fast_break = quick and dur < 7 and r.random() < 0.6
        base_tags = (["fromturnover"] if start_type == "turnover" else []) + (["fastbreak"] if fast_break else [])
        n_oreb = 0
        q_style = T["idx"] == STYLE_QUIRK[0] and sit[STYLE_QUIRK[1]]
        if dur > self.clock - 0.6:
            dur = self.clock - r.uniform(0.3, min(2.0, self.clock - 0.5))
        self.clock -= dur
        self.elapsed += dur
        while True:
            tags = base_tags + (["2ndchance"] if n_oreb else [])
            on = self.on[off]
            w = np.array([T["players"][i]["usage"] for i in on])
            if q_on and T["idx"] == 1 and 0 in on:
                w[on.index(0)] *= 2.2
            if q_on and T["idx"] == 6 and 1 in on:
                w[on.index(1)] *= 1.8
            fi = int(r.choice(on, p=w / w.sum()))
            P = T["players"][fi]
            if r.random() < 0.07:                            # non-shooting foul before the play develops
                in_pen = self.fouls[d] >= F.PENALTY_TEAM_FOULS
                self.fouls[d] += 1
                take = self.period >= 4 and self.clock < 60 and self.score[d] < self.score[off]   # trailing late: a deliberate foul
                self.ev("foul", "personal", d, int(r.choice(self.on[d])), **{F.FOUL_DRAWN_PERSON_ID: P["player_id"]},
                        qualifiers=["inpenalty"] if in_pen else [], descriptor="take" if take else "")
                if in_pen:
                    if self.free_throws(off, fi, 2, tags) or not self.rebound(off, 0.12):
                        return "free_throw" if self.actions[-1][F.ACTION_TYPE] == "freethrow" else "def_rebound"
                    n_oreb += 1
                    continue
                self.tick(2, 8)
                if self.clock <= 0.2:
                    self.clock = 0.0
                    return "period_start"
                continue
            pe = 0.5 * T["end_mix"] + 0.5 * P["end_mix"]
            if sit["def_in_penalty"]:
                pe = pe * np.array([1.0, 1.25, 1.0])
            if q_on and T["idx"] == 3:
                pe = pe * np.array([1.0, 1.0, 1.8])
            if q_on and T["idx"] == 4:
                pe = pe * np.array([1.0, 1.6, 1.0])
            end = ENDS[int(r.choice(3, p=pe / pe.sum()))]
            if end == "turnover":
                if r.random() < 0.06:
                    self.ev("turnover", "shot clock", off)
                    return "turnover"
                kind = str(r.choice(["bad pass", "lost ball", "traveling", "offensive foul", "out-of-bounds"], p=[0.38, 0.27, 0.12, 0.13, 0.10]))
                if kind == "offensive foul":
                    self.ev("foul", "offensive", off, fi, descriptor="charge")
                stealer = int(r.choice(self.on[d])) if kind in ("bad pass", "lost ball") and r.random() < 0.7 else None
                extra = {"stealPersonId": self.pid(d, stealer)} if stealer is not None else {}
                self.ev("turnover", kind, off, fi, **extra)
                if stealer is not None:
                    self.ev("steal", "", d, stealer)
                return "turnover"
            if end == "foul_drawn":
                self.fouls[d] += 1
                self.ev("foul", "personal", d, int(r.choice(self.on[d])), descriptor="shooting", **{F.FOUL_DRAWN_PERSON_ID: P["player_id"]})
                if self.free_throws(off, fi, 3 if r.random() < 0.06 else 2, tags):
                    return "free_throw"
                if not self.rebound(off, 0.12):
                    return "def_rebound"
                n_oreb += 1
                self.tick(1, 6)
                continue
            pm = 0.4 * T["shot_mix"] + 0.6 * P["shot_mix"]
            if q_on and T["idx"] == 0:
                pm = pm * np.array([1, 1, 1, 2.5, 1])
            if q_on and T["idx"] == 2:
                pm = pm * np.array([1.8, 1, 1, 1, 1])
            if q_on and T["idx"] == 5:
                pm = pm * np.array([1, 1, 1, 1, 1.7])
            fam = FAMS[int(r.choice(5, p=pm / pm.sum()))]
            is3 = fam in ("corner3", "above_break3")
            made = r.random() < MAKE_P[fam] * P["skill"]
            geo = self.shot_geometry(fam)
            sub = "Jump Shot" if is3 or fam == "midrange" or (fam == "paint" and r.random() < 0.5) else str(r.choice(["Layup", "Dunk", "Hook"], p=[0.7, 0.2, 0.1]))
            desc = self.descriptor(sub, fam, is3, P["arch"], n_oreb > 0, q_style)
            shot_tags = tags + (["pointsinthepaint"] if fam in ("rim", "paint") else [])
            if made:
                self.score[off] += 3 if is3 else 2
                extra = {}
                mates = [i for i in on if i != fi]
                if r.random() < ASSIST_P.get(desc, 0.4):
                    extra[F.ASSIST_PERSON_ID] = self.pid(off, int(r.choice(mates)))
                self.ev("3pt" if is3 else "2pt", sub, off, fi, shotResult="Made", descriptor=desc, qualifiers=shot_tags, **geo, **extra)
                if r.random() < (0.06 if fam in ("rim", "paint") else 0.012):   # and-one
                    self.fouls[d] += 1
                    self.ev("foul", "personal", d, int(r.choice(self.on[d])), descriptor="shooting", **{F.FOUL_DRAWN_PERSON_ID: P["player_id"]})
                    if self.free_throws(off, fi, 1, tags):
                        return "free_throw"
                    if not self.rebound(off, 0.10):
                        return "def_rebound"
                    n_oreb += 1
                    self.tick(1, 6)
                    continue
                return "made_basket"
            self.ev("3pt" if is3 else "2pt", sub, off, fi, shotResult="Missed", descriptor=desc, qualifiers=shot_tags, **geo)
            if r.random() < 0.07:
                self.ev("block", "", d, int(r.choice(self.on[d])))
            if self.clock <= 0.3:
                self.clock = 0.0
                return "period_start"                        # buzzer: no rebound logged
            if not self.rebound(off, 0.30 if fam in ("rim", "paint") else 0.22):
                return "def_rebound"
            n_oreb += 1
            self.tick(1, 6)

    # -- whole game ---------------------------------------------------------------
    def run(self) -> tuple[dict, dict]:
        r = self.rng
        first = self.home if r.random() < 0.5 else self.away
        while True:
            self.period += 1
            self.clock = F.REGULATION_PERIOD_SECONDS if self.period <= 4 else F.OVERTIME_PERIOD_SECONDS
            self.fouls = {t: 0 for t in self.T}
            self.period_tos = 0
            self.ev("period", "start")
            if self.period == 1:
                self.ev("jumpball", "recovered", first, self.on[first][4], descriptor=F.JUMPBALL_START)
            else:
                for team in (self.home, self.away):
                    self.set_lineup(team, self.pick_lineup(team, closing=self.period >= 4))
            off = first if self.period in (1, 4) or self.period > 4 else self.other[first]
            start = "period_start"
            while self.clock > 0.05:
                start = self.possession(off, start)
                if self.clock > 0.05:
                    scored_on = self.other[off] if start in ("made_basket", "free_throw") else None
                    self.dead_ball(scored_on)
                off = self.other[off]
            self.clock = 0.0
            self.ev("period", "end")
            if self.period >= 4 and self.score[self.home] != self.score[self.away]:
                break
        self.ev("game", "end")
        pbp = {F.SYNTHETIC_MARKER: True, "meta": {"version": 1, "note": "SYNTHETIC - not NBA data"},
               F.GAME: {F.GAME_ID: self.gid, F.ACTIONS: self.actions}}

        def side(tid: int) -> dict:
            t = self.T[tid]
            return {F.BOX_TEAM_ID: tid, F.BOX_TEAM_NAME: t["name"], F.BOX_TEAM_CITY: t["city"], F.BOX_TEAM_TRICODE: t["tricode"],
                    "score": self.score[tid],
                    F.BOX_PLAYERS: [{F.BOX_PLAYER_ID: p["player_id"], F.BOX_PLAYER_NAME: p["name"],
                                     F.BOX_PLAYER_STARTER: "1" if i < 5 else "0"} for i, p in enumerate(t["players"])]}
        box = {F.SYNTHETIC_MARKER: True, "meta": {"version": 1, "note": "SYNTHETIC - not NBA data"},
               F.GAME: {F.GAME_ID: self.gid, F.BOX_GAME_TIME_UTC: self.tip.strftime("%Y-%m-%dT%H:%M:%SZ"), "gameStatus": 3,
                        F.BOX_HOME: side(self.home), F.BOX_AWAY: side(self.away)}}
        return pbp, box


def _round_robin(n: int) -> list[list[tuple[int, int]]]:
    teams = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(teams[i], teams[n - 1 - i]) for i in range(n // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def write_synthetic_raw(root: Path, seed: int = 7) -> list[str]:
    """Write synthetic pbp/box files under root/raw/{pbp,box}/. Returns game ids. Idempotent per seed."""
    rng = np.random.default_rng(seed)
    league = make_league(rng)
    (root / "raw" / "pbp").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "box").mkdir(parents=True, exist_ok=True)
    assert len(list(combinations(range(len(TEAMS)), 2))) == 28
    ids = []
    for season, cycles in SEASONS.items():
        if season != min(SEASONS):
            league = drift(league, rng)
        day = datetime(season, 10, 22, tzinfo=timezone.utc)
        n = 0
        for c in range(cycles):
            if c == cycles // 2:
                league = drift(league, rng, team_sd=0.2)     # a mid-season shift for the tendency layer's drift term to measure
            for rnd in _round_robin(len(TEAMS)):
                for slot, (a, b) in enumerate(rnd):
                    h, w = (a, b) if (c + slot) % 2 == 0 else (b, a)
                    n += 1
                    gid = f"009{season % 100:02d}{n:05d}"
                    tip = day + timedelta(hours=23, minutes=30 * slot)
                    pbp, box = GameSim(gid, tip, league[h], league[w], rng).run()
                    (root / "raw" / "pbp" / f"{gid}.json").write_text(json.dumps(pbp))
                    (root / "raw" / "box" / f"{gid}.json").write_text(json.dumps(box))
                    ids.append(gid)
                day += timedelta(days=2)
    return ids
