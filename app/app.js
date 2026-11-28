// Tendency Board front end. Static: reads the contract JSON under data/ and nothing else.
// Every file is validated against its JSON Schema before it is rendered.
import { validate } from "./validate.js";

const DATA = "data/";
const SCHEMAS = ["manifest", "teams", "team_report", "games", "replay", "scorecard"];
const OPTIONAL = ["tendency_validation", "hot_hand"];      // contract 1.2; an older export has neither and still renders
const VIEWS = ["method", "team", "replay", "scorecard", "hothand", "ask"];
const ASK_KEY = "tendency-board.ask-url";                 // where the reader keeps the address of an ask service; nothing else is stored
const FAMS = ["rim", "paint", "midrange", "corner3", "above_break3"];
const HEADS = ["finisher", "ending_type", "shot_family"];      // the three the guess mode asks about; every export has them
const NUMBER = ["zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"];
const $ = (sel) => document.querySelector(sel);

const S = {
  schemas: {}, manifest: null, teams: null, games: null, scorecard: null, tendency_validation: null, hot_hand: null,
  teamId: null, reports: {}, gameId: null, replays: {}, idx: 0,
  guess: false, picks: {}, revealed: false, scored: new Set(),
  ask: { url: null, health: null, busy: false, question: "", last: null, problem: null, editing: false },
  tally: Object.fromEntries(HEADS.map((h) => [h, { n: 0, you: 0, model: 0, baseline: 0 }])),
};

// ---------- helpers ----------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const int = (n) => Number(n).toLocaleString("en-US");
const pct = (p, d = 0) => (p > 0 && p < 0.005 && d === 0 ? "<1%" : `${(p * 100).toFixed(d)}%`);
const pts = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(1)}`;
const seasonName = (y) => `${y}-${String((y + 1) % 100).padStart(2, "0")}`;       // seasons are stored by their start year
const signed = (v, d) => { const t = Math.abs(v).toFixed(d); return `${Number(t) === 0 ? "" : v > 0 ? "+" : "−"}${t}`; };     // never "−0.000"
const argmax = (a) => a.indexOf(Math.max(...a));
const periodName = (p) => (p <= 4 ? `Q${p}` : `OT${p - 4}`);
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const day = (iso) => new Date(iso).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
const team = (id) => S.teams.byId.get(id) || { tricode: String(id), name: String(id) };
const player = (id) => S.teams.players.get(id) || `Player ${id}`;
const head = (id) => S.manifest.heads.find((h) => h.id === id);
const isDecision = (id) => (head(id) || {}).kind === "decision";       // contract 1.5: a coach decision at a stoppage, not a possession outcome
const possHeads = () => S.manifest.heads.filter((h) => h.kind !== "decision");
const START = { period_start: "Start of period", made_basket: "After a made basket", def_rebound: "Off a defensive rebound", turnover: "Off a turnover", free_throw: "After free throws", other: "Other start" };
const SPLIT = { holdout_games: "Held-out games, same seasons as training", holdout_season: "Held-out season", prospective: "Games logged before tip-off" };

class ContractProblem extends Error {}

async function getJSON(path, schemaName) {
  let res;
  try { res = await fetch(DATA + path); } catch (e) { throw new Error(`Could not load ${path}. Check your connection and reload.`); }
  if (!res.ok) throw new Error(`Could not load ${path} (HTTP ${res.status}). Run the pipeline's export step, then reload.`);
  const obj = await res.json();
  if (schemaName) {
    const errors = validate(obj, S.schemas[schemaName]);
    if (errors.length) throw new ContractProblem(`${path} does not match ${schemaName}.schema.json:\n${errors.join("\n")}`);
  }
  return obj;
}

function showError(e) {
  const kind = e instanceof ContractProblem ? "The data does not match the contract, so nothing is shown" : "Something went wrong";
  $("#view").innerHTML = `<div class="error"><h3>${kind}</h3><pre>${esc(e.message)}</pre></div>`;
  console.warn(e.message);
}

// ---------- court diagram (zones are schematic, not to scale) ----------
function court(probs, hitFam, caption) {
  const max = Math.max(...probs, 0.01);
  const z = (i) => {
    const alpha = 0.05 + 0.85 * Math.min(1, probs[i] / Math.max(0.45, max));
    return { fill: `rgba(36,71,214,${alpha.toFixed(3)})`, light: alpha > 0.5, cls: `zone${FAMS[i] === hitFam ? " hit" : ""}` };
  };
  const label = (i, x, y, name) => {
    const c = z(i).light ? "#fff" : "#15181C";
    return `<text x="${x}" y="${y}" style="fill:${c}">${pct(probs[i])}</text><text class="zl" x="${x}" y="${y + 16}" style="fill:${c}">${name}${FAMS[i] === hitFam ? " ✓" : ""}</text>`;
  };
  const [rim, paint, mid, corner, ab3] = [0, 1, 2, 3, 4].map(z);
  return `<svg class="court" viewBox="0 0 500 380" role="img" aria-label="${esc(caption)}">
    <rect class="${ab3.cls}" x="0" y="0" width="500" height="380" fill="${ab3.fill}"/>
    <rect class="${corner.cls}" x="0" y="0" width="62" height="140" fill="#fff"/><rect class="${corner.cls}" x="0" y="0" width="62" height="140" fill="${corner.fill}"/>
    <rect class="${corner.cls}" x="438" y="0" width="62" height="140" fill="#fff"/><rect class="${corner.cls}" x="438" y="0" width="62" height="140" fill="${corner.fill}"/>
    <path d="M62,0 L62,140 A207.5,207.5 0 0 0 438,140 L438,0 Z" fill="#fff"/>
    <path class="${mid.cls}" d="M62,0 L62,140 A207.5,207.5 0 0 0 438,140 L438,0 Z" fill="${mid.fill}"/>
    <rect x="170" y="0" width="160" height="190" fill="#fff"/><rect class="${paint.cls}" x="170" y="0" width="160" height="190" fill="${paint.fill}"/>
    <circle cx="250" cy="52" r="46" fill="#fff"/><circle class="${rim.cls}" cx="250" cy="52" r="46" fill="${rim.fill}"/>
    <circle class="hoop" cx="250" cy="20" r="7"/><line class="hoop" x1="225" y1="10" x2="275" y2="10"/>
    ${label(0, 250, 58, "rim")}${label(1, 250, 140, "paint")}${label(2, 116, 120, "midrange")}
    ${label(3, 31, 66, "corner 3")}${label(4, 250, 318, "above-break 3")}
  </svg>`;
}

// ---------- probability rows ----------
function band(headId, split, p) {
  const h = S.scorecard.heads.find((x) => x.head === headId);
  const s = h && h.splits.find((x) => x.split === split);
  if (!s || !s.available) return null;
  const b = s.reliability.model.bins[Math.min(9, Math.floor(p * 10))];
  return b && b.n ? b : null;
}

function probRows(headId, labels, pred, hitIdx, split, pickIdx) {
  return `<div class="rows">${labels.map((lab, i) => {
    const pm = pred.model[i], pb = pred.baseline[i], b = band(headId, split, pm);
    const bandText = b
      ? `Held out, ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}% calls hit ${pct(b.obs_rate)} (n ${int(b.n)})`
      : "No held-out calls in this band yet";
    const mine = pickIdx === i ? `<span class="mine">your pick</span>` : "";
    return `<div class="prow${i === hitIdx ? " hit" : ""}">
      <span class="lab">${esc(lab)}${i === hitIdx ? ` <span class="ok" aria-label="what happened">✓</span>` : ""}${mine}</span>
      <span class="track" role="img" aria-label="model ${pct(pm)}, baseline ${pct(pb)}"><span class="fill" style="width:${(pm * 100).toFixed(1)}%"></span><span class="tick" style="left:calc(${(pb * 100).toFixed(1)}% - 1.5px)" title="baseline ${pct(pb)}"></span></span>
      <span class="p">${pct(pm)}</span>
      <span class="band">${bandText}. Baseline ${pct(pb)}.</span>
    </div>`;
  }).join("")}</div>`;
}

// Contract 1.4: a second model may be scored beside the trees. Older exports, and ones made without PyTorch, have none.
const MODEL_SHORT = { lightgbm: "Trees", transformer: "Transformer" };
const shortModel = (id) => MODEL_SHORT[id] || id;
function secondModel() {
  const models = S.scorecard.models || [], t = models.find((m) => m.model_id === "transformer");
  if (!t) return "";
  const won = S.scorecard.heads.filter((h) => h.model_id === "transformer").map((h) => head(h.head).label.toLowerCase());
  return ` A small transformer that reads the game's earlier possessions as a sequence was tried against the trees on the same possessions. ${won.length ? `It replaced them for: ${esc(won.join("; "))}.` : "It did not clearly beat them for any prediction, so it stays an experiment;"} <a href="#scorecard">the scorecard</a> shows both.`;
}

// ---------- view: method ----------
async function renderMethod() {
  const m = S.manifest, c = m.coverage, synth = m.data_kind === "synthetic";
  const g = S.games.games[0];
  let example = "";
  if (g) {
    try {
      const rep = await loadReplay(g.game_id);
      const p = rep.possessions.find((x) => x.eligible && x.actual.ending_type === "shot" && x.idx > 12) || rep.possessions.find((x) => x.eligible);
      if (p) {
        const b = band("shot_family", rep.split, Math.max(...p.pred.shot_family.model));
        example = `<figure>${court(p.pred.shot_family.model, p.actual.shot_family, "Example shot-location probabilities on a half court")}
          <figcaption>One held-out possession${synth ? " from the simulated sample" : ""}: ${esc(team(p.off_team_id).name)}, ${periodName(p.period)} ${clock(p.clock_s)}.
          Shading is the model's probability for each zone if the possession ends in a shot; the green outline is where the shot came from.
          ${b ? `Calls in the top zone's ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}% band came true ${pct(b.obs_rate)} of ${int(b.n)} times on held-out games.` : ""}</figcaption></figure>`;
      }
    } catch (e) { console.warn(e.message); }
  }
  const sp = m.splits;
  $("#view").innerHTML = `
  <section class="hero">
    <div>
      <h1>What happens on the next possession?</h1>
      <p class="lede">Tendency Board reads public NBA play-by-play and, before each possession, says who is likely to finish it, how it is likely to end, and where the shot is likely to come from. Then it checks itself against the dumbest reasonable guess and shows the result, win or lose.</p>
      <p><a class="btn primary" href="#replay" style="text-decoration:none;display:inline-block">Step through a game</a>
         <a class="btn" href="#scorecard" style="text-decoration:none;display:inline-block">See the scorecard</a></p>
    </div>
    ${example}
  </section>

  <div class="facts" aria-label="What this build covers">
    <div><b>${int(c.n_games)}</b><span>${synth ? "simulated " : ""}games, ${c.seasons.length} season${c.seasons.length > 1 ? "s" : ""}</span></div>
    <div><b>${int(c.n_possessions)}</b><span>possessions segmented</span></div>
    <div><b>${int(c.n_eligible)}</b><span>possessions the models are scored on or trained with</span></div>
    <div><b>${int(sp.holdout_games.n_games + sp.holdout_season.n_games)}</b><span>held-out games (${int(sp.holdout_games.n_games)} late-season, ${int(sp.holdout_season.n_games)} in a held-out season)</span></div>
  </div>
  ${synth ? `<p class="note">This build runs on simulated games so the whole thing can be clicked through before real data is ingested. Team and player names are placeholders. No number on any page is an NBA result, and no claim is made that the models beat their baselines on real games.</p>` : ""}
  ${m.field_names_verified ? "" : `<p class="note">The pipeline's feed field names have not been checked against a real game file yet. Until that check passes, treat the real-data path as untested.</p>`}

  <section class="section"><h2>${NUMBER[possHeads().length] || possHeads().length} predictions, each with a baseline to beat</h2>
    <div class="cols">${possHeads().map((h) => `<div class="sheet"><h3>${esc(h.label)}</h3>
      <p>${h.classes ? esc(h.class_labels.join(", ")) + "." : "One of the five offensive players on the floor."}${h.conditional_on ? ` Given that ${esc(h.conditional_on)}.` : ""}</p>
      <p class="small muted">Baseline: ${esc(h.baseline)}.${h.logged_before_tipoff === false ? " Scored looking back only; not part of the pregame log." : ""}</p></div>`).join("")}</div>
    <p style="margin-top:14px">The model for each prediction is gradient-boosted trees that start from the baseline's odds and learn corrections from game context: clock, score, fouls, timeouts, who is on the floor, how the possession started, and what has happened so far in the game. A prediction is kept only if it beats its baseline on real games it never saw; the scorecard shows by how much, and the margins are small.${secondModel()}</p>
  </section>

  ${m.labels && m.labels.length ? `<section class="section"><h2>What the feed lets us label</h2>
    <p>Beyond who finished and where the shot came from, play-by-play says a few things about how a possession went. Each is defined on only part of the possessions, and the table says how much. All of them are outcomes: they are counted and predicted, never used to predict the possession they describe.</p>
    <div class="sheet"><ul class="lab">
      ${m.labels.map((l) => `<li><div><b>${esc(l.label)}</b><div class="small muted">Read from: ${esc(l.source.charAt(0).toLowerCase() + l.source.slice(1))}. Defined when ${esc(l.defined_when)}.</div></div>
        <div class="cov"><b>${pct(l.coverage, 1)}</b><span class="small muted">${int(l.n_defined)} of ${int(l.n_possessions)} possessions</span></div></li>`).join("")}
    </ul>
    ${m.labels.filter((l) => l.note).map((l) => `<p class="small muted" style="margin-top:10px">${esc(l.note)}</p>`).join("")}
    <p class="note plain">A shot's style is what the scorer wrote down about the shot ("driving", "pullup", "cutting"). It is not the play that was called, and nothing here says what the defense did.</p></div>
  </section>` : ""}

  <section class="section"><h2>How it is kept honest</h2>
    <div class="cols">
      <div class="sheet"><h3>Held-out games, held-out season</h3><p>${esc(sp.rule)}</p>
        <p class="small muted">Training: ${int(sp.train.n_games)} games. Held out: ${int(sp.holdout_games.n_games)} games and ${int(sp.holdout_season.n_games)} games in a separate season.</p></div>
      <div class="sheet"><h3>Nothing from the future</h3><p>Season-to-date numbers for a game use only games that tipped off earlier. In-game counts use only earlier possessions. A test rewrites later outcomes and fails if any feature moves.</p></div>
      <div class="sheet"><h3>Looking back, until logged ahead</h3><p>Held-out results are computed after the games were played. The stricter test is a file written before tip-off that names a frozen model by its hash, scored after the game. ${S.scorecard.prospective && S.scorecard.prospective.available && S.scorecard.prospective.mode === "live" ? `${int(S.scorecard.prospective.n_games)} games have been logged that way so far.` : "No game has been logged that way yet; the scorecard says so."}</p></div>
      <div class="sheet"><h3>Uncertainty on every number</h3><p>Scorecard intervals resample whole games. Each probability in the replay sits beside how often calls like it came true on held-out games. Tendencies show an interval and a sample size, and are ranked by ${m.tendency_method.final ? "how probable it is that they are big enough to matter" : "the cautious end of the interval"}.</p></div>
    </div>
  </section>

  <section class="section"><h2>What play-by-play cannot see</h2>
    <div class="cols">
      <div class="sheet"><h3>No play calls, no coverages</h3><p>The feed records what ended a possession, not the set that was called or how the pick-and-roll was defended. Those need tracking or video. Nothing here predicts or implies them.</p></div>
      <div class="sheet"><h3>Lineups are reconstructed</h3><p>Who is on the floor comes from starters plus substitution events; where the source has no boxscore, starters are the first five players to act. ${pct(m.quality.lineup_ok_rate, 1)} of ${int(m.quality.n_possessions)} possessions reconstruct to five a side; the rest are left out of scoring.</p>
        <p class="small muted">${int(m.quality.n_heaves)} end-of-period heaves and ${int(m.quality.n_period_end)} possessions cut off by the horn are also left out. ${int(m.quality.n_anomalous_possessions)} changes of possession could not be explained by the segmenter.</p></div>
      <div class="sheet"><h3>${m.tendency_method.final ? "Tendencies are deviations, not rankings" : "Tendencies are a first pass"}</h3><p>${esc(m.tendency_method.label)}. ${esc(m.tendency_method.note)}</p></div>
    </div>
  </section>

  ${stoppagesSection()}

  ${defenseMethod()}

  ${summaryMethod()}

  ${askMethod()}

  <section class="section"><h2>Not built yet</h2>
    <p>${esc(m.not_built.join(", "))}. Teams with tracking and video vendors have richer labels than this; the point here is the method, not exclusive insight.</p>
  </section>`;
}

// ---------- view: team report ----------
// ---------- method: the dead-ball table and the coach decisions read off it (contract 1.5) ----------
function stoppagesSection() {
  const m = S.manifest, st = m.stoppages, dec = m.heads.filter((h) => h.kind === "decision");
  if (!st || !dec.length) return "";
  const perGame = (n) => (n / st.n_games).toFixed(1);
  const chk = st.personal_foul_check, to = st.timeouts;
  const rowsFor = Object.fromEntries(st.decision_rows.map((r) => [r.head, r]));
  return `<section class="section"><h2>What coaches did at stoppages</h2>
    <p>Substitutions, timeouts and closing lineups are decided when play stops, not during a possession. So beside the possession table there is a second one: ${int(st.n_stoppages)} stoppages in ${int(st.n_games)} games, each with the clock, the score, how long every player on the floor has been out there and how many fouls he has, the timeouts each team has used, the run in progress, and what each bench did before play resumed. Everything in this section describes what coaches did. None of it says what they should have done.</p>
    <div class="cols">${dec.map((h) => `<div class="sheet"><h3>${esc(h.label)}</h3>
      <p>${esc(h.class_labels.join(" or "))}. Asked when ${esc(h.conditional_on)}.</p>
      <p class="small muted">Baseline: ${esc(h.baseline)}. ${rowsFor[h.id] ? `${int(rowsFor[h.id].n)} ${esc(h.unit)} here, ${pct(rowsFor[h.id].n_yes / Math.max(1, rowsFor[h.id].n), 1)} of them a yes.` : ""} Scored looking back only.</p></div>`).join("")}</div>
    <div class="cols" style="margin-top:18px">
      <div class="sheet"><h3>What counts as a stoppage</h3>
        <div class="scroll"><table class="tv fit"><thead><tr><th>What stopped play</th><th>Rows</th><th>A game</th></tr></thead><tbody>
          ${st.kinds.filter((k) => k.n).map((k) => `<tr><th>${esc(k.label)}</th><td>${int(k.n)}</td><td>${perGame(k.n)}</td></tr>`).join("")}</tbody></table></div>
        <p class="small muted">A made basket or a made last free throw is a row because a timeout can be called before the inbound, though a substitution cannot. The run in progress is the points each team scored in the last ${st.run_possessions} possessions, and the points scored since the other team last did.</p></div>
      <div class="sheet"><h3>What could be checked</h3>
        <p>${chk.n ? `The feed prints each player's running foul count on every foul. This table's own count matches it on ${int(chk.agree)} of ${int(chk.n)} fouls.` : "This feed does not print running foul counts, so the table's own count could not be checked against it."}</p>
        <p class="small muted">Of ${int(to.n_full)} full timeouts, ${int(to.n_when_mandatory_due)} came when the rules stop play anyway (none yet in the quarter with 6:59 left, or fewer than two with 2:59 left). The feed does not say whether a coach asked for those, so they are not decisions here. Another ${int(to.n_live_ball)} were called with the ball in play, where there is no matching "did not call one" to compare with.</p></div>
      <div class="sheet"><h3>Decisions left alone</h3>
        <ul class="leftalone">${st.not_modelled.map((x) => `<li><b>${esc(x.label)}</b> <span class="small muted">(${int(x.n)} in this data)</span><br><span class="small">${esc(x.why)}</span></li>`).join("")}</ul></div>
    </div>
  </section>`;
}

// ---------- method: the defense proxy and what was left out of it (contract 1.6) ----------
function defenseMethod() {
  const tv = S.tendency_validation, dc = tv && tv.defense;
  if (!dc) return "";                                  // an export from before contract 1.6
  const units = dc.slices.filter((x) => x.scope !== "defense_team");
  const better = units.filter((x) => x.methods.find((m) => m.method === x.shipped).verdict === "ahead").length;
  const pickedBetter = units.some((x) => x.picked && x.picked.verdict === "ahead");
  return `<section class="section"><h2>Defense: a proxy from outcomes</h2>
    <div class="cols">
      <div class="sheet"><h3>What a team report shows</h3><p>Seven things about opponents' possessions against a defense: where their shots came from (rim, paint, midrange, corner three, above-the-break three), how often they drew a foul, how often they turned it over. Each is set beside what the same opponents did against every other defense, so the row is about the defense and not about its schedule.</p></div>
      <div class="sheet"><h3>What it cannot say</h3><p>Play-by-play shows how a possession ended and which five were on the floor. It does not show where defenders stood or who guarded whom. So this cannot identify a pick-and-roll coverage or any other scheme, and no row here names one. That takes tracking or video.</p></div>
      <div class="sheet"><h3>Tested and left out</h3><p>The same rows were built for single five-man lineups and for the lineups that include one player. To be shown, such a row has to predict the second half of a season better than its team's own row. On the real seasons that decision was made on, it did not, so no report shows them. ${dc.available && units.length ? `The check is rerun on every export. Here, on ${seasonName(dc.season)}, it came out better in ${better} of ${units.length} comparisons${pickedBetter ? "" : ", and the few rows a report would have singled out did no better"}.` : "There is too little data here to rerun the check."} <a href="#scorecard">The numbers are on the scorecard.</a></p>
        <p class="small muted">Not in this build: who guarded whom (per-game matchup data exists but is not read here) and league season-level defensive statistics.</p></div>
    </div>
  </section>`;
}

// ---------- team report: what the bench did (contract 1.5) ----------
function benchSection(r) {
  const cd = r.coach_decisions;
  if (!cd) return "";
  const lvl = Math.round(cd.interval_level * 100);
  const row = (x) => {
    const away = x.n > 0 && (x.interval[0] > x.league_rate || x.interval[1] < x.league_rate);
    return `<tr><th>${esc(x.label)}</th><td>${x.n ? `${int(x.k)} of ${int(x.n)}` : "none yet"}</td>
      <td><b>${pct(x.rate, 1)}</b>${away ? ` <span class="tag">${x.rate > x.league_rate ? "above" : "below"} league</span>` : ""}<br><span class="small muted">${pct(x.interval[0], 1)} to ${pct(x.interval[1], 1)}</span></td>
      <td>${pct(x.league_rate, 1)}<br><span class="small muted">n = ${int(x.league_n)}</span></td></tr>`;
  };
  const tbl = (first, rows) => `<div class="scroll"><table class="tv fit"><thead><tr><th>${first}</th><th>Count</th><th>Rate, ${lvl}% interval</th><th>League</th></tr></thead><tbody>${rows.map(row).join("")}</tbody></table></div>`;
  const c = cd.closing, mins = Math.round(c.window_s / 60);
  return `<div class="sheet"><h2 style="margin-bottom:4px">What the bench did</h2>
    <p class="muted" style="margin-top:0">${esc(seasonName(cd.season))}. ${esc(cd.note)}</p>
    <h3>Timeouts</h3>
    <p class="small muted">Stoppages in regulation where calling one was a choice: the team had a timeout, and the rules were not about to stop play anyway.</p>
    ${tbl("When play stopped", cd.timeouts)}
    <h3 style="margin-top:22px">After a foul</h3>
    <p class="small muted">A player on the floor is whistled; taken out means before the clock ran again.</p>
    ${tbl("The foul", cd.foul_trouble)}
    <h3 style="margin-top:22px">Who closes</h3>
    ${c.n_close_games ? `<p class="small muted">${int(c.n_close_games)} game${c.n_close_games === 1 ? "" : "s"} within ${c.close_margin} points with ${mins} minutes left. A closer is one of the five who played most of those minutes. Counted over the close games in which the player had been on the floor before the fourth quarter.</p>
    <div class="scroll"><table class="tv fit"><thead><tr><th>Player</th><th>Closed</th><th>Rate, ${lvl}% interval</th></tr></thead><tbody>
      ${c.players.map((x) => `<tr><th>${esc(player(x.player_id))}</th><td>${int(x.k)} of ${int(x.n)}</td><td><b>${pct(x.rate)}</b><br><span class="small muted">${pct(x.interval[0])} to ${pct(x.interval[1])}</span></td></tr>`).join("")}</tbody></table></div>`
    : `<p class="note plain">No game this season was within ${c.close_margin} points with ${mins} minutes left.</p>`}
  </div>`;
}

// ---------- team report: defense, a proxy from outcomes (contract 1.6) ----------
function defenseCheck() {
  const tv = S.tendency_validation;
  return tv && tv.defense && tv.defense.available ? tv.defense : null;
}

function defenseSection(r) {
  const d = r.defense;
  if (!d) return "";
  const lvl = Math.round(d.interval_level * 100);
  const maxAbs = Math.max(0.01, ...d.rows.flatMap((x) => x.interval.map(Math.abs))) * 1.08;
  const X = (v) => (150 + (v / maxAbs) * 142).toFixed(1);
  const rows = d.rows.map((x) => {
    const up = x.deviation >= 0, col = !x.excludes_zero ? "#6E7580" : up ? "#2447D6" : "#C8253A";
    const unit = x.outcome.head === "shot_family" ? "opponent shots" : "opponent possessions";
    return `<li class="${x.excludes_zero ? "" : "weak"}">
      <div class="what"><b>${esc(x.outcome.label)}</b>
        <div class="nums">${pct(x.raw_rate, 1)} on ${int(x.n)} ${unit}; ${pct(x.expected_rate, 1)} expected from the same opponents.</div></div>
      <div><span class="dev ${up ? "up" : "down"}">${pts(x.deviation)} pts</span>
        <span class="small muted"> ${lvl}% interval ${pts(x.interval[0])} to ${pts(x.interval[1])}${x.excludes_zero ? "" : ", includes zero"}</span>
        <svg class="ivl" viewBox="0 0 300 34" role="img" aria-label="Difference ${pts(x.deviation)} points, interval ${pts(x.interval[0])} to ${pts(x.interval[1])}, n ${x.n}">
          <line class="axis" x1="4" y1="17" x2="296" y2="17"/><line class="zero" x1="150" y1="4" x2="150" y2="30"/>
          <line class="bar" x1="${X(x.interval[0])}" y1="17" x2="${X(x.interval[1])}" y2="17" stroke="${col}"/>
          <circle class="dot" cx="${X(x.deviation)}" cy="17" r="5.5" stroke="${col}"/></svg></div></li>`;
  }).join("");
  const dc = defenseCheck();
  const held = dc && dc.coverage.find((c) => c.scope === "defense_team");
  const ahead = dc ? dc.slices.filter((x) => x.scope === "defense_team").map((x) => ({ x, m: x.methods.find((m) => m.method === x.shipped) })) : [];
  const short = { shot_profile: "shot locations", foul_drawn: "fouls drawn", turnover: "turnovers" };
  const said = { ahead: "better", level: "no better", behind: "worse" };
  const list = (a) => (a.length > 1 ? `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}` : a[0]);
  const byVerdict = ["ahead", "level", "behind"].map((v) => [v, ahead.filter(({ m }) => m.verdict === v).map(({ x }) => short[x.outcomes])]).filter(([, a]) => a.length);
  const checked = ahead.length ? ` Checked on the ${seasonName(dc.season)} season, fitted on its first half and asked about each game of the second. Against the opponents' habits alone, these rows predicted ${byVerdict.map(([v, a]) => `${list(a)} ${said[v]}`).join(", ")}${held ? `. The ${lvl}% intervals held ${pct(held.coverage_with_drift ?? held.coverage, 1)} of the second half's differences` : ""}. <a href="#scorecard">The table is on the scorecard.</a>` : "";
  return `<div class="sheet"><h2 style="margin-bottom:4px">Defense: what opponents did against this team</h2>
    <p class="muted" style="margin-top:0">${esc(seasonName(d.season))}. ${int(d.n_possessions)} opponent possessions in ${int(d.n_games)} games.</p>
    <p>Each row sets what opponents did against this defense beside what was expected of them. Expected: ${esc(d.baseline.charAt(0).toLowerCase() + d.baseline.slice(1))}. So a defense gets no credit for a schedule of poor offenses, and none for fast breaks its own offense did or did not give away. The difference is pulled toward zero by an amount set from how much the league's defenses really differ.</p>
    ${d.rows.length ? `<ul class="tend">${rows}</ul>` : `<p class="note plain">Not enough opponent possessions yet for a row.</p>`}
    <p class="note plain">${esc(d.note)} Intervals include the opponents' own uncertainty${d.drift_included ? " and how much these differences moved between the two halves of the season before" : "; they do not allow for a defense changing, because there is no earlier season here to measure that on"}.${checked}</p>
  </div>`;
}

// ---------- team report: the written summary (contract 1.7) ----------
// The small number after a sentence is the row of the table it was written from and checked against.
function summarySection(r) {
  const s = r.summary;
  if (!s) return "";                                   // an export from before contract 1.7, or one built without summaries
  const llm = s.generated_by.kind === "llm";
  const ref = (rows) => `<sup aria-label="from row ${rows.join(" and ")}">${rows.join(", ")}</sup>`;
  const n = r.tendencies.length;
  const body = llm
    ? `<p class="sum">${s.sentences.map((x) => `${esc(x.text)}${ref(x.rows)}`).join(" ")}</p>`
    : `<ul class="sum">${s.sentences.map((x) => `<li>${esc(x.text)}${ref(x.rows)}</li>`).join("")}</ul>`;
  const label = llm
    ? `Machine-written by ${esc(s.generated_by.model)} from the ${n} rows below and nothing else. It was not told which team this is, and players reached it as placeholders. A program then checked each sentence against the row it cites: every figure and name in it is in that row.`
    : `Filled in from the first ${s.sentences.length === 1 ? "row" : `${s.sentences.length} rows`} below by a fixed template. No language model wrote this.${s.attempts ? " A machine-written draft was discarded because it did not match the table." : ""}`;
  return `<div class="summary"><h3>Summary</h3>${body}<p class="small muted">${label} <a href="#method">How that works.</a></p></div>`;
}

function summaryMethod() {
  const b = S.manifest.summaries;
  if (!b) return "";
  const share = (k, n) => `${int(k)} of ${int(n)}`;
  const here = b.n_llm
    ? `In this export ${share(b.n_llm, b.n_reports)} summaries are machine-written (${esc(b.models.join(", "))}). The check discarded the first draft for ${share(b.n_first_draft_rejected, b.n_asked)} teams, and ${b.n_fell_back === 0 ? "none" : int(b.n_fell_back)} ended on the template.`
    : `In this export no language model was run: all ${int(b.n_reports)} summaries are the templated ones, and each says so.`;
  return `<section class="section"><h2>The written summary</h2>
    <div class="cols">
      <div class="sheet"><h3>Written from the table, and only the table</h3><p>The paragraph above a team's tendencies is written by a language model that is sent that team's table and nothing else. It is not told which team it is, and a player reaches it as a placeholder that is swapped back afterwards. It has nothing to look anything up with and nothing to remember.</p></div>
      <div class="sheet"><h3>Checked sentence by sentence</h3><p>Each sentence names the row it was written from. A program reads it against that row: every number has to appear there as printed, every name has to be in it, the direction has to match, and a difference has to come with its interval or its sample size. One failure and the whole summary is thrown away.</p></div>
      <div class="sheet"><h3>What is shown when it fails</h3><p>Three sentences filled in from the first three rows by a fixed template, labeled as that. ${here}</p>
        <p class="small muted">The check cannot judge whether a sentence chose the most useful rows or reads well. It can only say that nothing in it disagrees with the table.</p></div>
    </div>
  </section>`;
}

// ---------- view: ask (contract 1.8). The page works without it; it needs a service whose address the reader sets ----------
function askMethod() {
  const a = S.manifest.ask;
  if (!a) return "";                                   // an export from before contract 1.8
  return `<section class="section"><h2>The ask box</h2>
    <div class="cols">
      <div class="sheet"><h3>A question becomes one lookup</h3><p>A question typed into <a href="#ask">Ask</a> is turned into one query over the tendency table: a team or a player, a situation, an outcome, a season, and what to set it against. The translator can return only values from fixed lists and a player's name. The query it ran is always shown.</p></div>
      <div class="sheet"><h3>The sentence is filled in, not written</h3><p>A program runs the query and fills a fixed sentence from the rows it gets, with each rate's interval and sample size. No words from a language model are shown, so there is nowhere for something remembered about a team to get in.</p>
        <p class="small muted">The table behind it has ${int(a.n_cells)} rows over ${esc(a.seasons.join(", "))}; ${int(a.n_estimated)} have enough possessions for an estimate, and the rest are listed with their counts.</p></div>
      <div class="sheet"><h3>What it will not answer</h3><p>Play calls, coverages, injuries, forecasts and anything outside the table get one fixed sentence that says the data cannot answer and why. A question the table could hold but has no row for gets the same.</p>
        <p class="small muted">It runs on a small service of its own. Without one this site is unchanged, and Ask says how to start it.</p></div>
    </div>
  </section>`;
}

const askUrl = () => { try { return localStorage.getItem(ASK_KEY); } catch (e) { return null; } };
const setAskUrl = (u) => { try { if (u) localStorage.setItem(ASK_KEY, u); else localStorage.removeItem(ASK_KEY); } catch (e) { /* private mode: kept for this visit only */ } S.ask.url = u; };

async function askHealth() {
  const a = S.ask;
  a.health = null;
  try {
    const res = await fetch(`${a.url}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    a.health = await res.json();
    a.problem = null;
  } catch (e) { a.problem = `No ask service answered at ${a.url}. Start it, or change the address.`; }
}

async function askSend(question) {
  const a = S.ask;
  a.busy = true; a.question = question; a.last = null; a.problem = null;
  renderAskBody();
  try {
    const res = await fetch(`${a.url}/ask`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question }) });
    const obj = await res.json().catch(() => null);
    if (!res.ok) throw new Error(obj && typeof obj.detail === "string" ? obj.detail : `The service answered HTTP ${res.status}.`);
    const errors = validate(obj, S.schemas.ask_response);
    if (errors.length) throw new Error(`The service's reply does not match ask_response.schema.json, so it is not shown:\n${errors.join("\n")}`);
    a.last = obj;
  } catch (e) { a.problem = e instanceof TypeError ? `No ask service answered at ${a.url}. Start it, or change the address.` : e.message; }
  a.busy = false;
  renderAskBody();
}

function askResult(o) {
  const lvl = Math.round(o.interval_level * 100);
  const by = o.translated_by.kind === "llm" ? `by ${esc(o.translated_by.model)}, which returned the query and no words` : "by keyword rules, with no language model";
  const kinds = o.data_kind !== S.manifest.data_kind ? `<p class="note">The service is reading ${esc(o.data_kind)} data. The rest of this site is showing ${esc(S.manifest.data_kind)} data.</p>` : "";
  if (o.status === "refused" || o.status === "failed" || (o.status === "no_row" && !o.query))
    return `${kinds}<div class="answer no"><p class="sum">${esc(o.answer)}</p>
      <p class="small muted">${o.status === "refused" ? `A fixed sentence. The question was read ${by}, and nothing was looked up.` : "Nothing was looked up."}</p></div>`;
  const q = o.query, r0 = o.rows[0];
  const CMP = { baseline: "Against its own baseline", league: "Ranked across the league", team: `Against ${q.compare_to}`, season: `Against ${q.compare_to}` };
  const fields = [["Team", q.team || "Any"], ["Player", q.player || "None"], ["Situation", r0 ? r0.situation_label : q.situation],
    ["Outcome", r0 ? r0.outcome_label : q.outcome], ["Season", q.season], ["Compared", CMP[q.comparison]]];
  const who = (r) => (r.scope === "player" ? `${esc(r.player)} <span class="muted">${esc(r.team)}</span>` : esc(r.team_name));
  const p1 = (x) => `${(x * 100).toFixed(1)}%`;
  const ranked = o.rows.some((r) => r.rank !== undefined);
  const dev = (r) => (r.estimated ? `${pts(r.deviation)} <span class="muted">(${pts(r.interval[0])} to ${pts(r.interval[1])})</span>` : `<span class="muted">too few to estimate</span>`);
  const rate = (r) => (r.estimated ? `${p1(r.rate)} <span class="muted">(${p1(r.rate_interval[0])} to ${p1(r.rate_interval[1])})</span>` : `<span class="muted">none</span>`);
  const rows = o.rows.map((r) => `<tr>${ranked ? `<td>${r.rank}</td>` : ""}<th scope="row" class="l">${who(r)}</th><td>${esc(r.season)}</td>
    <td>${p1(r.raw_rate)} <span class="muted">${int(r.k)} of ${int(r.n)} ${esc(r.unit)}</span></td>
    <td>${r.estimated ? p1(r.baseline_rate) : ""}</td><td>${dev(r)}</td><td>${rate(r)}</td></tr>`).join("");
  const table = o.rows.length ? `<div class="scroll"><table class="tv fit"><thead><tr>${ranked ? "<th>Rank</th>" : ""}<th class="l">Row</th><th>Season</th><th>Observed</th><th>Baseline</th>
      <th>Difference in points, ${lvl}% interval</th><th>Pooled rate, ${lvl}% interval</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${o.n_rows > o.rows.length ? `<p class="small muted">${int(o.rows.length)} of ${int(o.n_rows)} rows shown.</p>` : ""}` : "";
  return `${kinds}<div class="answer${o.status === "answered" ? "" : " no"}"><p class="sum">${esc(o.answer)}</p>
      <p class="small muted">Filled in from the ${o.rows.length === 1 ? "row" : "rows"} below by a fixed template. No language model wrote this sentence. The question was turned into the query below ${by}.</p></div>
    <h3>The query it ran</h3>
    <dl class="query">${fields.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
    ${o.rows.length ? `<h3>The rows it got</h3>${table}` : ""}
    <details><summary>The SQL</summary><pre class="sql">${esc(o.sql.text)}\n${esc(JSON.stringify(o.sql.params))}</pre></details>`;
}

function renderAskBody() {
  const a = S.ask, box = $("#askbody");
  if (!box) return;
  const m = S.manifest.ask;
  if (!S.schemas.ask_response) {
    box.innerHTML = `<div class="sheet"><h3>This export has no ask box</h3><p>The data on this page was exported before contract 1.8, so it carries no schema for the ask service's replies. Export again, then reload.</p></div>`;
    return;
  }
  if (!a.url || a.editing) {
    box.innerHTML = `<div class="sheet"><h3>${a.url ? "Change the service address" : "The ask box needs its service"}</h3>
      <p>Every other page here is a static file. This one sends your question to a small service that turns it into a query, runs it, and sends back the rows. Nothing is sent anywhere until you set an address below.</p>
      <p class="small muted">To run it on this machine: <code>python -m tendency serve --root data/synthetic</code>, then use <code>http://127.0.0.1:8000</code>.</p>
      <form id="askset" class="askform"><label class="sr" for="askurl">Service address</label>
        <input id="askurl" type="url" inputmode="url" placeholder="http://127.0.0.1:8000" value="${esc(a.url || "")}" autocomplete="off" spellcheck="false">
        <button class="btn primary" type="submit">Save</button>${a.url ? `<button class="btn" type="button" id="askforget">Forget it</button>` : ""}</form>
      ${a.problem ? `<p class="note">${esc(a.problem)}</p>` : ""}</div>`;
    $("#askset").addEventListener("submit", async (e) => {
      e.preventDefault();
      const u = $("#askurl").value.trim().replace(/\/+$/, "");
      if (!/^https?:\/\/[^\s/]+/.test(u)) { a.problem = "An address starts with http:// or https://."; renderAskBody(); return; }
      setAskUrl(u); a.editing = false; a.last = null;
      await askHealth();
      if (a.problem) a.editing = true;
      renderAskBody();
    });
    const f = $("#askforget");
    if (f) f.addEventListener("click", () => { setAskUrl(null); a.editing = false; a.health = null; a.last = null; a.problem = null; renderAskBody(); });
    return;
  }
  const h = a.health;
  const serving = h ? `${esc(h.data_kind)} data, ${esc(h.seasons.join(", "))}; questions are read ${h.translated_by.kind === "llm" ? `by ${esc(h.translated_by.model)}` : "by keyword rules, with no language model"}` : "not checked yet";
  const examples = m && m.examples.length ? `<p class="small muted">Try one:</p><div class="pick" id="askex">${m.examples.map((x, i) => `<button type="button" data-i="${i}">${esc(x)}</button>`).join("")}</div>` : "";
  box.innerHTML = `<form id="askform" class="askform"><label class="sr" for="askq">Your question</label>
      <input id="askq" type="text" maxlength="${h ? h.max_question : 300}" placeholder="How often do they get to the rim after a timeout?" value="${esc(a.question)}" autocomplete="off">
      <button class="btn primary" type="submit" ${a.busy ? "disabled" : ""}>${a.busy ? "Asking" : "Ask"}</button></form>
    ${examples}
    <div id="askout" aria-live="polite">${a.busy ? `<p class="loading">Looking it up.</p>` : a.problem ? `<div class="error"><h3>No answer</h3><pre>${esc(a.problem)}</pre></div>` : a.last ? askResult(a.last) : ""}</div>
    <p class="small muted askfoot">Service at ${esc(a.url)}: ${serving}. <button class="link" type="button" id="askchange">Change</button></p>`;
  $("#askform").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#askq").value.trim(); if (v && !a.busy) askSend(v); });
  const ex = $("#askex");
  if (ex) ex.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !a.busy) askSend(m.examples[Number(b.dataset.i)]); });
  $("#askchange").addEventListener("click", () => { a.editing = true; a.problem = null; renderAskBody(); });
}

async function renderAsk() {
  const a = S.ask;
  if (a.url === null) a.url = askUrl();
  $("#view").innerHTML = `<div class="pagehead"><div><h1>Ask</h1>
    <p class="muted">A question becomes one lookup in the tendency table. The answer is filled in from the rows it finds, with each rate's interval and sample size.</p></div></div>
    <div id="askbody"></div>`;
  renderAskBody();
  if (a.url && !a.health && S.schemas.ask_response) { await askHealth(); if (a.problem) a.editing = true; renderAskBody(); }
}

async function renderTeam() {
  const teams = S.teams.teams;
  if (!S.teamId) S.teamId = teams[0].team_id;
  const t = team(S.teamId);
  if (!S.reports[S.teamId]) S.reports[S.teamId] = await getJSON(`${S.manifest.files.team_reports_dir}/${S.teamId}.json`, "team_report");
  const r = S.reports[S.teamId];
  const lvl = Math.round(r.interval_level * 100);
  const maxAbs = Math.max(0.01, ...r.tendencies.flatMap((x) => x.interval.map(Math.abs))) * 1.08;
  const X = (v) => (150 + (v / maxAbs) * 142).toFixed(1);
  const UNIT = { shot_family: "shots", action_style: "shots", creation: "made baskets", fast_break: "possessions ending in a shot or at the line" };
  const unit = (x) => (x.scope === "player" ? "possessions on the floor" : UNIT[x.outcome.head] || "possessions");
  const versus = { league_same_situation: "the league rate", team_overall_plus_league_shift: "expected from the team's overall rate and the league-wide shift in this situation", player_overall: "his share across all situations",
    team_elsewhere_plus_league_shift: "expected from the team's rate in its other possessions and the league-wide shift in this situation",
    player_lineup_adjusted: "what his usage and the four teammates on the floor predict" };
  // Contract 1.2 rows carry the probability they were ranked by. Older exports do not, and keep the older wording.
  const ranked = r.threshold !== undefined && r.tendencies.every((x) => x.prob_beyond !== undefined);
  const thr = ranked ? (r.threshold * 100).toFixed(0) : null;
  const point = (n) => `${n} point${n === "1" ? "" : "s"}`;
  const rows = r.tendencies.map((x, i) => {
    const up = x.deviation >= 0, col = !x.excludes_zero ? "#6E7580" : up ? "#2447D6" : "#C8253A";
    return `<li class="${x.excludes_zero ? "" : "weak"}">
      <div class="what">${r.summary ? `<span class="rowno" aria-label="Row ${i + 1}">${i + 1}</span>` : ""}<b>${esc(x.outcome.label)}</b><br><span class="sit">${esc(x.situation.label)}</span>
        <div class="nums">${pct(x.raw_rate, 1)} observed on ${int(x.n)} ${unit(x)}; ${pct(x.baseline_rate, 1)} is ${versus[x.baseline_kind]}.</div></div>
      <div><span class="dev ${up ? "up" : "down"}">${pts(x.deviation)} pts</span>
        <span class="small muted"> ${lvl}% interval ${pts(x.interval[0])} to ${pts(x.interval[1])}${x.excludes_zero ? "" : ", includes zero"}</span>
        ${ranked ? `<div class="small muted">${pct(x.prob_beyond)} likely to be more than ${point(thr)} ${up ? "above" : "below"} baseline</div>` : ""}
        <svg class="ivl" viewBox="0 0 300 34" role="img" aria-label="Deviation ${pts(x.deviation)} points, interval ${pts(x.interval[0])} to ${pts(x.interval[1])}, n ${x.n}">
          <line class="axis" x1="4" y1="17" x2="296" y2="17"/><line class="zero" x1="150" y1="4" x2="150" y2="30"/>
          <line class="bar" x1="${X(x.interval[0])}" y1="17" x2="${X(x.interval[1])}" y2="17" stroke="${col}"/>
          <circle class="dot" cx="${X(x.deviation)}" cy="17" r="5.5" stroke="${col}"/></svg></div></li>`;
  }).join("");
  const flagged = r.tendencies.filter((x) => x.excludes_zero).length;
  const doubtful = ranked ? r.tendencies.reduce((a, x) => a + 1 - x.prob_beyond, 0) : 0;
  const tv = S.tendency_validation && S.tendency_validation.available ? S.tendency_validation : null;
  const sel = tv && tv.selection.find((x) => x.method === tv.shipped);
  const checked = sel ? ` Checked on the ${seasonName(tv.season)} season: fitted on its first half, the ${lvl}% intervals of the rows each team would have been shown held ${pct(sel.coverage_with_drift ?? sel.coverage, 1)} of what the second half produced (${int(sel.n_rows)} rows). <a href="#scorecard">The full check is on the scorecard.</a>` : "";
  const intro = ranked
    ? `A tendency is how far a rate sits from its baseline, in percentage points. Small samples are pulled toward the baseline, by an amount set from how much teams and players really differ. Rows are ordered by how likely the tendency is to be more than ${point(thr)} in its direction, so a big estimate on a thin sample does not jump the queue.`
    : `${r.method.final ? "" : "This is a stand-in until the partial-pooling model is fitted, and it is labeled that way on purpose. "}A tendency is how far a rate sits from its baseline, in percentage points. Small samples are pulled toward the baseline. Rows are ordered by the end of the interval nearest zero, so a big estimate on a thin sample does not jump the queue.`;
  const outro = ranked
    ? `These ${r.tendencies.length} rows were picked from ${int(r.n_candidates)} candidates. Going by each row's own probability, about ${doubtful.toFixed(1)} of the ${r.tendencies.length} shown ${r.tendencies.length === 1 ? "is" : "are"} expected to be smaller than ${point(thr)} or to point the other way. Intervals include the baseline's own uncertainty${r.drift_included ? " and how much tendencies moved between the two halves of the season before, so they describe the games to come and not only the ones played" : "; they do not allow for tendencies changing, because there is no earlier season here to measure that on"}. Player rows are measured against what his usage and the four teammates on the floor predict.${checked}`
    : `These ${r.tendencies.length} rows were picked from ${int(r.n_candidates)} candidates, and ${flagged} have a ${lvl}% interval clear of zero. With that many candidates, roughly ${Math.round(r.n_candidates * (1 - r.interval_level))} would clear zero by chance alone. Intervals treat the baseline as exact, which makes them a little narrow. Player rows do not adjust for who else is on the floor.`;
  $("#view").innerHTML = `
  <div class="pagehead"><div><h2>Team report: ${esc(t.name)}</h2>
    <p class="muted">${int(r.n_possessions)} offensive possessions in ${int(r.n_games)} games, through ${day(r.as_of_utc)}.</p></div>
    <div class="pick" role="group" aria-label="Team">${teams.map((x) => `<button data-team="${x.team_id}" aria-pressed="${x.team_id === S.teamId}" title="${esc(x.name)}">${esc(x.tricode)}</button>`).join("")}</div></div>
  <div class="sheet">
    ${summarySection(r)}
    <p><strong>${esc(r.method.label)}.</strong> ${intro}</p>
    ${r.tendencies.length ? `<ul class="tend">${rows}</ul>` : `<p class="note plain">No situation has at least 25 possessions for this team yet. Ingest more games and run the export again.</p>`}
    <p class="note plain">${outro}</p>
  </div>
  ${defenseSection(r)}
  ${benchSection(r)}`;
  document.querySelectorAll("[data-team]").forEach((b) => b.addEventListener("click", () => { S.teamId = Number(b.dataset.team); render(); }));
}

// ---------- view: replay ----------
async function loadReplay(gameId) {
  if (!S.replays[gameId]) S.replays[gameId] = await getJSON(`${S.manifest.files.replays_dir}/${gameId}.json`, "replay");
  return S.replays[gameId];
}

const STYLE_PHRASE = { plain_jumper: "a jump shot", drive: "a driving shot", pullup: "a pull-up", running: "a running shot", floater: "a floater",
  turnaround_hook: "a turnaround or hook", cut_lob: "a cut or lob finish", step_back: "a step-back", putback: "a putback", plain_finish: "a layup or dunk" };

// Contract 1.3 fields on what happened; absent in older exports, where this returns "".
function actualExtra(p) {
  const a = p.actual, bits = [];
  if (a.assisted) bits.push(`Assisted by ${player(a.assister_id)}.`);
  else if (a.assisted === false) bits.push("Unassisted.");
  if (typeof a.secs_to_first_action === "number") bits.push(`First action ${a.secs_to_first_action.toFixed(0)} s into the possession${a.tempo ? ` (${a.tempo})` : ""}.`);
  if (a.fast_break) bits.push("Tagged a fast break by the scorer.");
  return bits.length ? ` ${bits.join(" ")}` : "";
}

function actualText(p) {
  const a = p.actual, who = a.finisher_id ? player(a.finisher_id) : team(p.off_team_id).name;
  const shot = STYLE_PHRASE[a.action_style] || "a shot";
  const fam = { rim: "at the rim", paint: "in the paint", midrange: "from midrange", corner3: "from the corner, for three", above_break3: "from above the break, for three", heave: "on a heave" }[a.shot_family] || "";
  const extra = `${a.n_off_reb ? ` after ${a.n_off_reb} offensive rebound${a.n_off_reb > 1 ? "s" : ""}` : ""}. ${a.points} point${a.points === 1 ? "" : "s"} on the possession.`;
  if (a.ending_type === "shot") return `${who} ${a.shot_made ? "made" : "missed"} ${shot} ${fam}${a.and_one ? ", and drew the foul" : ""}${extra}`;
  if (a.ending_type === "foul_drawn") return `${who} drew a foul and went to the line${extra}`;
  if (a.ending_type === "turnover") return `${a.finisher_id ? who + " turned it over" : "Team turnover"}${extra}`;
  if (a.ending_type === "period_end") return "The period ended before a shot, foul or turnover.";
  return "The play-by-play shows a change of possession the segmenter could not explain.";
}

function scoreGuess(key, p) {
  if (S.scored.has(key)) return;
  S.scored.add(key);
  const a = p.actual, truth = { finisher: a.finisher_id ? p.off_lineup.indexOf(a.finisher_id) : -1, ending_type: head("ending_type").classes.indexOf(a.ending_type), shot_family: a.ending_type === "shot" ? FAMS.indexOf(a.shot_family) : -1 };
  for (const h of HEADS) {
    if (truth[h] < 0 || S.picks[h] === undefined) continue;
    const t = S.tally[h];
    t.n += 1;
    t.you += S.picks[h] === truth[h] ? 1 : 0;
    t.model += argmax(p.pred[h].model) === truth[h] ? 1 : 0;
    t.baseline += argmax(p.pred[h].baseline) === truth[h] ? 1 : 0;
  }
}

function tallyTable() {
  const any = HEADS.some((h) => S.tally[h].n);
  if (!any) return `<p class="small muted">Pick who finishes and how it ends, then show the model. Your hits are counted against the model's top pick and the baseline's top pick on the same possessions.</p>`;
  const cell = (k, n) => (n ? `${k} of ${n}` : "none yet");
  return `<table class="tally"><thead><tr><th>Top pick right</th><th>You</th><th>Model</th><th>Baseline</th></tr></thead><tbody>
    ${HEADS.map((h) => { const t = S.tally[h]; return `<tr><td>${esc(head(h).label)}</td><td>${cell(t.you, t.n)}</td><td>${cell(t.model, t.n)}</td><td>${cell(t.baseline, t.n)}</td></tr>`; }).join("")}
  </tbody></table><p class="small muted">A few dozen possessions is far too few to separate you from the model. The scorecard uses thousands.</p>`;
}

async function renderReplay() {
  const games = S.games.games;
  if (!games.length) { $("#view").innerHTML = `<div class="sheet"><h2>Replay</h2><p>No held-out games were exported. Run the pipeline on a longer date range.</p></div>`; return; }
  if (!S.gameId) S.gameId = games[0].game_id;
  const rep = await loadReplay(S.gameId);
  const n = rep.possessions.length;
  S.idx = Math.max(0, Math.min(n - 1, S.idx));
  const p = rep.possessions[S.idx], key = `${S.gameId}:${p.idx}`;
  const home = team(rep.home_team_id), away = team(rep.away_team_id), off = team(p.off_team_id);
  const hiding = S.guess && p.eligible && !S.revealed && !S.scored.has(key);
  const a = p.actual, endClasses = head("ending_type").classes;
  const hit = { finisher: a.finisher_id ? p.off_lineup.indexOf(a.finisher_id) : -1, ending_type: endClasses.indexOf(a.ending_type), shot_family: a.ending_type === "shot" ? FAMS.indexOf(a.shot_family) : -1 };
  const labels = { finisher: p.off_lineup.map(player), ending_type: head("ending_type").class_labels, shot_family: head("shot_family").class_labels };
  // Heads added after the first three (contract 1.3): shown when this export predicts them, never part of the guess.
  const extraHeads = S.manifest.heads.filter((h) => !HEADS.includes(h.id) && p.pred && p.pred[h.id]);
  for (const h of extraHeads) { labels[h.id] = h.class_labels; hit[h.id] = a[h.id] ? h.classes.indexOf(a[h.id]) : -1; }
  const option = (h, i, lab) => `<button data-pick="${h}" data-i="${i}" aria-pressed="${S.picks[h] === i}">${esc(lab)}</button>`;

  const panel = (h, extraTop = "") => {
    const hd = head(h);
    let body;
    if (!p.eligible) body = `<p class="muted">No prediction: ${esc(p.ineligible_reason)}.</p>`;
    else if (hiding && !HEADS.includes(h)) body = `<p class="muted">Shown with the model. Not part of the guess.</p>`;
    else if (hiding) body = `<div class="opts" role="group" aria-label="Your pick: ${esc(hd.label)}">${labels[h].map((l, i) => option(h, i, l)).join("")}</div>${h === "shot_family" ? `<p class="small muted" style="margin-top:8px">Optional. Counted only if the possession ends in a shot.</p>` : ""}`;
    else body = `${extraTop}${probRows(h, labels[h], p.pred[h], hit[h], rep.split, S.guess ? S.picks[h] : undefined)}`;
    return `<div class="sheet"><h3>${esc(hd.label)}</h3>${hd.conditional_on ? `<p class="small muted" style="margin:0">If ${esc(hd.conditional_on)}.</p>` : ""}${body}</div>`;
  };
  const canReveal = S.picks.finisher !== undefined && S.picks.ending_type !== undefined;

  $("#view").innerHTML = `
  <div class="pagehead"><div><h2>Game replay</h2>
    <p class="muted">${SPLIT[rep.split]}. The models never trained on this game.</p></div>
    <label class="small">Game<br><select class="btn" id="game">${games.map((g) => `<option value="${g.game_id}" ${g.game_id === S.gameId ? "selected" : ""}>${day(g.tipoff_utc)}: ${esc(team(g.away.team_id).tricode)} ${g.away.score} at ${esc(team(g.home.team_id).tricode)} ${g.home.score}</option>`).join("")}</select></label></div>

  <div class="scorebug" aria-label="Score before this possession">
    <div class="side away ${p.off_team_id === rep.away_team_id ? "has" : ""}"><span>${esc(away.tricode)}</span><span>${p.away_score}</span></div>
    <div class="mid"><b>${periodName(p.period)} ${clock(p.clock_s)}</b>${esc(off.tricode)} ball</div>
    <div class="side ${p.off_team_id === rep.home_team_id ? "has" : ""}"><span>${p.home_score}</span><span>${esc(home.tricode)}</span></div>
  </div>
  <div class="stepper">
    <button class="btn" id="prev" ${S.idx === 0 ? "disabled" : ""}>Previous</button>
    <input type="range" id="scrub" min="0" max="${n - 1}" value="${S.idx}" aria-label="Possession">
    <button class="btn primary" id="next" ${S.idx === n - 1 ? "disabled" : ""}>Next</button>
    <span class="small muted">Possession ${S.idx + 1} of ${n}</span>
  </div>
  <ul class="ctx" aria-label="Situation before the possession">
    <li>${START[p.start_type]}</li>
    ${p.after_timeout ? `<li class="on">After a timeout</li>` : ""}${p.def_in_penalty ? `<li class="on">Defense in the penalty</li>` : ""}
    <li>${p.home_score === p.away_score ? "Tied" : `${esc(off.tricode)} ${(p.off_team_id === rep.home_team_id) === (p.home_score > p.away_score) ? "up" : "down"} ${Math.abs(p.home_score - p.away_score)}`}</li>
  </ul>

  <div class="sheet" style="margin-bottom:18px">
    <div class="guessbar">
      <label class="switch"><input type="checkbox" id="guess" ${S.guess ? "checked" : ""}> Guess before the model</label>
      ${hiding ? `<button class="btn primary" id="reveal" ${canReveal ? "" : "disabled"}>Show the model</button>` : ""}
      ${HEADS.some((h) => S.tally[h].n) ? `<button class="btn" id="reset">Reset my tally</button>` : ""}
      <ul class="key"><li><i class="k-model"></i>model</li><li><i class="k-base"></i>baseline</li><li><i class="k-actual"></i>what happened</li></ul>
    </div>
    ${S.guess ? tallyTable() : ""}
    <div aria-live="polite">${hiding ? `<div class="outcome none">The outcome and the model's numbers are hidden until you pick.</div>` : `<div class="outcome${p.eligible ? "" : " none"}">${esc(actualText(p) + actualExtra(p))}</div>`}</div>
  </div>

  <div class="heads">
    ${panel("finisher")}${panel("ending_type")}
    ${panel("shot_family", p.eligible && !hiding ? court(p.pred.shot_family.model, a.ending_type === "shot" ? a.shot_family : null, "Model probability by shot zone") : "")}
    ${extraHeads.map((h) => panel(h.id)).join("")}
  </div>
  ${hiding ? `<p style="margin-top:16px"><button class="btn primary" id="reveal2" ${canReveal ? "" : "disabled"}>Show the model</button> <span class="small muted">${canReveal ? "" : "Pick a finisher and an ending first."}</span></p>` : ""}`;

  const go = (i) => { S.idx = i; S.picks = {}; S.revealed = false; render(); };
  $("#prev").addEventListener("click", () => go(S.idx - 1));
  $("#next").addEventListener("click", () => go(S.idx + 1));
  $("#scrub").addEventListener("change", (e) => go(Number(e.target.value)));
  $("#game").addEventListener("change", (e) => { S.gameId = e.target.value; go(0); });
  $("#guess").addEventListener("change", (e) => { S.guess = e.target.checked; S.picks = {}; S.revealed = false; render(); });
  document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => {
    const h = b.dataset.pick, i = Number(b.dataset.i);
    if (S.picks[h] === i) delete S.picks[h]; else S.picks[h] = i;
    render();
  }));
  for (const rv of [$("#reveal"), $("#reveal2")]) if (rv) rv.addEventListener("click", () => { scoreGuess(key, p); S.revealed = true; render(); });
  const rs = $("#reset");
  if (rs) rs.addEventListener("click", () => { for (const h of HEADS) S.tally[h] = { n: 0, you: 0, model: 0, baseline: 0 }; S.scored.clear(); render(); });
}

// ---------- view: scorecard ----------
function reliabilityPlot(rel, title) {
  const x = (v) => 38 + v * 210, y = (v) => 218 - v * 210;
  const maxN = Math.max(1, ...rel.model.bins.map((b) => b.n));
  const marks = (bins, model) => bins.filter((b) => b.n).map((b) => {
    const se = Math.sqrt((b.obs_rate * (1 - b.obs_rate)) / b.n), lo = Math.max(0, b.obs_rate - 1.96 * se), hi = Math.min(1, b.obs_rate + 1.96 * se);
    const cx = x(b.mean_pred).toFixed(1), col = model ? "#2447D6" : "#6E7580";
    const whisker = `<line x1="${cx}" x2="${cx}" y1="${y(lo).toFixed(1)}" y2="${y(hi).toFixed(1)}" stroke="${col}" stroke-width="1.5"/>`;
    const tip = `<title>${model ? "Model" : "Baseline"}: predicted ${pct(b.mean_pred, 1)}, observed ${pct(b.obs_rate, 1)}, n = ${int(b.n)}</title>`;
    if (model) return `${whisker}<circle cx="${cx}" cy="${y(b.obs_rate).toFixed(1)}" r="${(3 + 4 * Math.sqrt(b.n / maxN)).toFixed(1)}" fill="${col}" stroke="#fff" stroke-width="1.5">${tip}</circle>`;
    return `${whisker}<rect x="${(cx - 4).toFixed(1)}" y="${(y(b.obs_rate) - 4).toFixed(1)}" width="8" height="8" fill="#fff" stroke="${col}" stroke-width="2">${tip}</rect>`;
  }).join("");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${y(0)}" y2="${y(1)}"/><line class="grid" x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}"/><text x="${x(v)}" y="234" text-anchor="middle">${v * 100}</text><text x="32" y="${y(v) + 4}" text-anchor="end">${v * 100}</text>`).join("");
  return `<svg class="rel" viewBox="0 0 260 256" role="img" aria-label="${esc(title)}">${grid}
    <line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>${marks(rel.baseline.bins, false)}${marks(rel.model.bins, true)}
    <text x="143" y="251" text-anchor="middle">predicted probability, %</text><text x="10" y="113" text-anchor="middle" transform="rotate(-90 10 113)">observed, %</text></svg>`;
}

function renderScorecard() {
  const sc = S.scorecard, lvl = Math.round(sc.interval_level * 100), synth = sc.data_kind === "synthetic";
  const V = { model_better: "Model ahead", baseline_better: "Baseline ahead", inconclusive: "No clear difference" };
  const f = (m) => `<td>${m.log_loss.toFixed(4)}</td><td>${m.interval[0].toFixed(4)} to ${m.interval[1].toFixed(4)}</td>`;
  const block = (h, s) => {
    if (!s.available) return `<div class="splitname"><h3>${SPLIT[s.split]}</h3></div><p class="note plain">Not available: ${esc(s.reason)}.</p>`;
    const sign = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(4)}`;
    const bins = (r) => r.bins.filter((b) => b.n).map((b) => `<tr><td>${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%</td><td>${pct(b.mean_pred, 1)}</td><td>${pct(b.obs_rate, 1)}</td><td>${int(b.n)}</td></tr>`).join("");
    return `<div class="splitname"><h3>${SPLIT[s.split]}</h3><span class="verdict ${s.verdict}">${V[s.verdict]}</span></div>
    <div class="sc"><div>
      <table><thead><tr><th>Log-loss, lower is better</th><th>Value</th><th>${lvl}% interval</th></tr></thead><tbody>
        <tr><th>Model</th>${f(s.model)}</tr><tr><th>Baseline</th>${f(s.baseline)}</tr>
        <tr><th>Model minus baseline</th><td>${sign(s.delta.log_loss)}</td><td>${sign(s.delta.interval[0])} to ${sign(s.delta.interval[1])}</td></tr></tbody></table>
      <p class="small muted" style="margin-top:10px">${int(s.n_possessions)} ${esc(head(h.head).unit || "possessions")} in ${int(s.n_games)} games. ${isDecision(h.head) ? "" : `Top pick right: model ${pct(s.top1.model, 1)}, baseline ${pct(s.top1.baseline, 1)} (n = ${int(s.top1.n)}).`}
        Average calibration gap: model ${pct(s.reliability.model.ece, 1)}, baseline ${pct(s.reliability.baseline.ece, 1)}.</p>
      <details><summary>Calibration table</summary>
        <table><thead><tr><th>Model band</th><th>Mean predicted</th><th>Observed</th><th>n</th></tr></thead><tbody>${bins(s.reliability.model)}</tbody></table>
        <table><thead><tr><th>Baseline band</th><th>Mean predicted</th><th>Observed</th><th>n</th></tr></thead><tbody>${bins(s.reliability.baseline)}</tbody></table></details>
    </div><div>${reliabilityPlot(s.reliability, `Calibration for ${head(h.head).label}, ${SPLIT[s.split]}`)}
      <ul class="key" style="justify-content:center"><li><i class="k-model" style="width:10px;border-radius:50%"></i>model</li><li><i style="width:9px;height:9px;border:2px solid #6E7580"></i>baseline</li><li>bars: ${lvl}% range of the observed rate</li></ul></div></div>`;
  };
  // Prospective results are optional in the contract (added in 1.1). A missing block and an empty one read the same.
  const pro = sc.prospective || { available: false, mode: null, reason: "No predictions have been logged before tip-off yet", n_dates: 0, n_games: 0, n_late: 0, models: [], heads: [] };
  const replayed = pro.mode === "replay";
  const proHtml = pro.available
    ? `<p class="prohead"><b>${int(pro.n_games)}</b> game${pro.n_games === 1 ? "" : "s"} on <b>${int(pro.n_dates)}</b> date${pro.n_dates === 1 ? "" : "s"}, ${day(pro.first_date)} to ${day(pro.last_date)}.
        ${replayed ? "" : "Each game has a file written before tip-off that names the frozen model by hash; the commit time of that file is the proof."}
        ${pro.n_late ? ` ${int(pro.n_late)} more ${pro.n_late === 1 ? "was" : "were"} logged after tip-off and ${pro.n_late === 1 ? "is" : "are"} left out.` : ""}</p>
      ${replayed ? `<p class="note"><strong>Replay, not prospective evidence.</strong> These dates were replayed day by day from games already on disk: freeze a model, write the pregame files, then score. It shows the logging and scoring work. It does not show the model was fixed before the games were played.</p>` : ""}
      <p class="small muted">Frozen model${pro.models.length === 1 ? "" : "s"}: ${pro.models.map((m) => `${esc(m.id)} (trained through ${day(m.trained_through)}, sha256 ${esc(m.sha256.slice(0, 12))})`).join("; ")}.
        Each game is scored with pregame numbers from the morning of the game, so it cannot see another game from the same night.</p>
      ${pro.heads.map((h) => `<h3 class="prohd">${esc(head(h.head).label)}</h3>${block(h, h.result)}`).join("")}`
    : `<p class="note plain">${esc(pro.reason || "Nothing logged yet")}. Everything under "Looking back" was computed after the games were played, on games the models never trained on. It becomes prospective only when a prediction file is committed before tip-off and scored afterwards.</p>`;
  const sheet = (h) => `<div class="sheet"><h2 style="margin-bottom:4px">${esc(head(h.head).label)}</h2>
    <p class="small muted">Baseline: ${esc(head(h.head).baseline)}.${head(h.head).conditional_on ? ` Scored only when ${esc(head(h.head).conditional_on)}.` : ""}${head(h.head).logged_before_tipoff === false ? " Not part of the pregame log: these results are looking back only." : ""}</p>
    ${h.splits.map((s) => block(h, s)).join("")}${threeWay(h, lvl)}</div>`;
  $("#view").innerHTML = `
  <div class="pagehead"><div><h2>Scorecard</h2>
    <p class="muted">Each prediction against its baseline on games the models never saw. A verdict needs the whole ${lvl}% interval of the difference on one side of zero.</p></div></div>
  ${synth ? `<p class="note">These numbers come from simulated games. "Model ahead" here shows the evaluation runs end to end; it is not evidence the model beats its baseline on NBA data.</p>` : ""}
  <section class="sheet pro" aria-label="Prospective results"><h2 style="margin-bottom:4px">Logged before tip-off</h2>${proHtml}</section>
  <h2 class="lookback">Looking back</h2>
  <p class="muted" style="margin-top:0">Retrospective: held-out games and a held-out season, scored after the fact.</p>
  ${modelsNote()}
  ${sc.heads.filter((h) => !isDecision(h.head)).map(sheet).join("")}
  ${sc.heads.some((h) => isDecision(h.head)) ? `<h2 class="lookback">Coach decisions</h2>
  <p class="muted" style="margin-top:0">What a bench did at a stoppage, scored the same way: can it be anticipated better than a table of league rates? This says nothing about whether the decision was a good one. Trees only; rows are stoppages, fouls or players, not possessions.</p>
  ${sc.heads.filter((h) => isDecision(h.head)).map(sheet).join("")}` : ""}
  <p class="note plain">Intervals: ${esc(sc.bootstrap)}. Calibration pools every class: each dot is a band of predicted probabilities and how often those calls came true.</p>
  ${tendencyCheck()}
  ${defenseCheckTable()}`;
}

// ---------- scorecard: two models on the same possessions (contract 1.4) ----------
function modelsNote() {
  const models = S.scorecard.models || [], t = models.find((m) => m.fit);
  if (models.length < 2 || !t) return "";
  const f = t.fit, mins = f.seconds >= 90 ? `${(f.seconds / 60).toFixed(0)} minutes` : `${f.seconds.toFixed(0)} seconds`;
  return `<div class="sheet"><h3>Two models, the same possessions</h3>
    <p>Each prediction below is scored for ${models.map((m) => `<b>${esc(shortModel(m.model_id).toLowerCase())}</b> (${esc(m.label.charAt(0).toLowerCase() + m.label.slice(1))})`).join(" and ")}. ${esc(t.note)} ${t.experimental ? "It is marked experimental." : ""} The replay shows each prediction's primary model.</p>
    <p class="small muted">The transformer: ${f.configs.length} configurations tried and no more; "${esc(f.chosen)}" kept (${esc(f.selection)}). ${int(f.n_parameters)} parameters, ${int(f.n_train_games)} training games, ${mins} for all ${f.configs.length} on ${esc(f.hardware)}. It is given the same inputs and starts from the same baselines as the trees.</p>
    <details><summary>The configurations</summary><div class="scroll"><table class="tv"><thead><tr><th>Configuration</th><th>Width</th><th>Layers</th><th>Reads earlier possessions</th><th>Parameters</th><th>Epochs</th><th>Seconds</th><th>Validation loss</th></tr></thead><tbody>
      ${f.configs.map((c) => `<tr class="${c.chosen ? "ship" : ""}"><th>${esc(c.name)}${c.chosen ? ` <span class="tag">kept</span>` : ""}</th><td>${c.d_model}</td><td>${c.n_layers}</td><td>${c.history ? "yes" : "no"}</td><td>${int(c.n_parameters)}</td><td>${c.epochs}</td><td>${c.seconds.toFixed(0)}</td><td>${c.valid_loss === null ? "none" : c.valid_loss.toFixed(4)}</td></tr>`).join("")}
    </tbody></table></div><p class="small muted">Validation loss is the sum of the five predictions' log-losses on the validation games. "no_history" is the same network with the sequence switched off.</p></details></div>`;
}

function threeWay(h, lvl) {
  const ch = (h.challengers || [])[0];
  if (!ch) return "";
  const prim = h.model_id || "lightgbm", P = shortModel(prim), C = shortModel(ch.model_id);
  const m = (x) => (x ? `${x.log_loss.toFixed(4)}<br><span class="small muted">${x.interval[0].toFixed(4)} to ${x.interval[1].toFixed(4)}</span>` : "none");
  const V = { ahead: `${C} ahead`, level: "No clear difference", behind: `${P} ahead` };
  const cls = { ahead: "model_better", level: "inconclusive", behind: "model_better" };     // model against model: a clear winner is green whichever it is
  const rows = ch.splits.map((c) => {
    const s = h.splits.find((x) => x.split === c.split);
    if (!c.available || !s || !s.available) return `<tr><th>${SPLIT[c.split]}</th><td colspan="5">Not available: ${esc(c.reason || (s && s.reason) || "no held-out possessions")}.</td></tr>`;
    return `<tr><th>${SPLIT[c.split]}<br><span class="small muted">${int(c.n_possessions)} possessions, ${int(c.n_games)} games</span></th><td>${m(s.baseline)}</td><td>${m(s.model)}</td><td>${m(c.model)}</td>
      <td>${signed(c.vs_primary.log_loss, 4)}<br><span class="small muted">${signed(c.vs_primary.interval[0], 4)} to ${signed(c.vs_primary.interval[1], 4)}</span></td>
      <td><span class="verdict small ${cls[c.verdict]}">${V[c.verdict]}</span></td></tr>`;
  }).join("");
  const exp = (S.scorecard.models || []).find((x) => x.model_id === ch.model_id);
  return `<div class="splitname"><h3>Three ways: baseline, ${P.toLowerCase()}, ${C.toLowerCase()}</h3>${exp && exp.experimental ? `<span class="tag">experimental</span>` : ""}</div>
    <div class="scroll"><table class="tv"><thead><tr><th>Held out</th><th>Baseline</th><th>${P}</th><th>${C}</th><th>${C} minus ${P.toLowerCase()} (${lvl}% interval)</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="small muted">Log-loss with its ${lvl}% interval; the difference is paired on the same possessions. ${C === "Transformer" ? `The transformer replaces the trees for this prediction only if the difference is clear of zero on both rows${ch.second_seed ? ", and again when it is trained a second time with a different random seed" : ""}.` : "The transformer replaced the trees for this prediction: it was ahead on both rows, and again when trained a second time with a different random seed."}
      ${ch.second_seed ? `Second training run: ${ch.second_seed.map((s) => `${SPLIT[s.split].toLowerCase()}, ${signed(s.vs_primary.log_loss, 4)} (${signed(s.vs_primary.interval[0], 4)} to ${signed(s.vs_primary.interval[1], 4)}), ${V[s.verdict].toLowerCase()}`).join("; ")}.` : ""}</p>`;
}

// ---------- scorecard: does the tendency layer predict the half-season it has not seen? ----------
function tendencyCheck() {
  const v = S.tendency_validation;
  if (!v) return "";                                   // an export from before contract 1.2: nothing to show, nothing claimed
  const head2 = `<h2 class="lookback">Team-report tendencies, checked</h2>
    <p class="muted" style="margin-top:0">The team report is fitted on the first half of a season and asked to predict the second, against two simpler ways of doing the same job.</p>`;
  if (!v.available) return `${head2}<div class="sheet"><p class="note plain">Not available: ${esc(v.reason || "too little data")}.</p></div>`;
  const lvl = Math.round(v.interval_level * 100), V = { ahead: "Ahead of the stand-in", level: "Level with the stand-in", behind: "Behind the stand-in" };
  const cls = { ahead: "model_better", level: "inconclusive", behind: "baseline_better" };
  const what = { team: "Team rows", player: "Player rows" };
  const drift = v.drift.available;
  const slice = (s) => {
    const ship = s.methods.find((m) => m.method === v.shipped);
    return `<div class="splitname"><h3>${what[s.scope]}</h3>${ship && ship.verdict ? `<span class="verdict ${cls[ship.verdict]}">${V[ship.verdict]}</span>` : ""}</div>
    <div class="scroll"><table class="tv"><thead><tr><th>Fitted on the first half</th><th>Log-loss, second half</th><th>Against the stand-in (95% interval)</th>
      <th>${lvl}% intervals held</th>${drift ? `<th>With the drift term</th>` : ""}<th>Interval width${drift ? ", with drift" : ""}</th></tr></thead><tbody>
      ${s.methods.map((m) => { const d = m.with_drift || m, vs = d.vs_stand_in; return `<tr class="${m.method === v.shipped ? "ship" : ""}"><th>${esc(m.label)}${m.method === v.shipped ? ` <span class="tag">in the report</span>` : ""}</th>
        <td>${d.log_loss.toFixed(5)}</td><td>${m.method === "stand_in" ? "the reference" : `${signed(vs.diff, 6)} (${signed(vs.interval[0], 6)} to ${signed(vs.interval[1], 6)})`}</td>
        <td>${pct(m.coverage, 1)}</td>${drift ? `<td>${pct(m.with_drift.coverage, 1)}</td>` : ""}<td>${pts(d.interval_width).slice(1)} pts</td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="small muted">${int(s.n_cells)} cells (one ${s.scope}, one situation, one outcome) with at least ${v.min_n} trials in each half; ${int(s.n_trials)} second-half trials. Every method is scored on the same cells.${drift ? " Log-loss and width are with the drift term added." : ""}</p>`;
  };
  const sdPts = (sd) => (sd * 0.1875 * 100).toFixed(1);          // a logit-scale sd read at a 25% rate, in percentage points
  const rateDrift = v.drift.rate.map((d) => `${d.scope} rates ${d.sd.toFixed(3)} (about ${sdPts(d.sd)} points at a 25% rate; ${int(d.n_cells)} cells)`).join(", ");
  const control = v.control.map((c) => `${pct(c.coverage, 1)} of ${int(c.n_cells)} ${c.scope} cells`).join(" and ");
  const ship = v.selection.find((x) => x.method === v.shipped), old = v.selection.find((x) => x.method === "stand_in");
  const selText = (x) => `${pct(x.coverage, 1)}${x.coverage_with_drift !== null ? ` as fitted and ${pct(x.coverage_with_drift, 1)} with the drift term` : ""} (${int(x.n_rows)} rows)`;
  return `${head2}
  <div class="sheet">
    <p>Season ${seasonName(v.season)}: ${int(v.n_games_first)} games to fit on, ${int(v.n_games_second)} to predict. Log-loss scores each method's rate for a cell against what the second half did (lower is better). "Intervals held" is how often a ${lvl}% interval for the second-half count contained the count; a calibrated method scores about ${lvl}%.</p>
    ${v.slices.map(slice).join("")}
    <div class="splitname"><h3>Why the intervals needed a drift term</h3></div>
    <ul class="plainlist">
      ${control ? `<li>Split the same season by alternating games instead of by date, so no time passes between the halves, and raw-rate intervals hold ${control}. The shortfall above is teams and players changing, not the interval arithmetic.</li>` : ""}
      ${drift ? `<li>How much they change was measured on the ${seasonName(v.drift.season)} season (${int(v.drift.n_games)} games), never on the season being scored: the spread between halves that binomial noise does not explain, on the logit scale, is ${rateDrift}.</li>
      <li>The deviation a report row shows moves less than the rate itself: ${v.drift.deviation.map((d) => `${esc(d.label.toLowerCase())}, ${d.sd.toFixed(3)} (${int(d.n_cells)} cells)`).join("; ")}.</li>` : `<li>There is no earlier season in this export to measure drift on, so the intervals here do not include it.</li>`}
    </ul>
    <div class="splitname"><h3>The rows a report would have shown</h3></div>
    <p>Picking each team's top ten from a couple of hundred candidates favours rows that got lucky. ${ship ? `For the top ten the report's method would have shown, ${lvl}% intervals held ${selText(ship)}.` : ""} ${old ? `For the top ten the old stand-in would have shown, ${selText(old)}.` : ""}</p>
    ${v.deviation.length ? `<p class="small muted">The interval drawn on a report row is for the deviation from baseline. Against the second half's deviation it held: ${v.deviation.map((d) => `${esc(d.label.toLowerCase())}, ${pct(d.coverage_with_drift ?? d.coverage, 1)} of ${int(d.n_cells)}`).join("; ")}.</p>` : ""}
    <p class="note plain">Log-loss interval: ${esc(v.bootstrap)}. A verdict needs that whole interval on one side of zero.</p>
  </div>`;
}

// ---------- scorecard: the defense proxy, checked (contract 1.6) ----------
function defenseCheckTable() {
  const tv = S.tendency_validation, v = tv && tv.defense;
  if (!v) return "";
  const head2 = `<h2 class="lookback">Defense rows, checked</h2>
    <p class="muted" style="margin-top:0">A team report's defense rows are fitted on the first half of a season and asked for every opponent possession of the second.</p>`;
  if (!v.available) return `${head2}<div class="sheet"><p class="note plain">Not available: ${esc(v.reason || "too little data")}.</p></div>`;
  const lvl = Math.round(v.interval_level * 100), cls = { ahead: "model_better", level: "inconclusive", behind: "baseline_better" };
  const short = { shot_profile: "Where shots came from", foul_drawn: "Fouls drawn", turnover: "Turnovers" };
  const diff = (d) => `${signed(d.diff, 6)} (${signed(d.interval[0], 6)} to ${signed(d.interval[1], 6)})`;
  const get = (x, id) => x.methods.find((m) => m.method === id);
  const team = v.slices.filter((x) => x.scope === "defense_team"), units = v.slices.filter((x) => x.scope !== "defense_team");
  const V = { ahead: "Ahead", level: "Level", behind: "Behind" };
  const unitName = { defense_player: "Lineups with one player", defense_lineup: "One five-man lineup" };
  const cov = (scope) => v.coverage.find((c) => c.scope === scope);
  const tc = cov("defense_team");
  return `${head2}
  <div class="sheet">
    <p>Season ${seasonName(v.season)}: ${int(v.n_games_first)} games to fit on, ${int(v.n_games_second)} to predict. The reference is what tonight's opponent usually does, with no defense in it. A defense row earns its place only if adding it predicts better than that (lower log-loss).</p>
    <div class="splitname"><h3>Team rows</h3><span class="tag">in the report</span></div>
    <div class="scroll"><table class="tv"><thead><tr><th>Opponents' possessions</th><th>Opponents' habits alone</th><th>With the defense row</th><th>Difference (95% interval)</th><th>Against the unadjusted rate</th></tr></thead><tbody>
      ${team.map((x) => { const r = get(x, x.reference), m = get(x, x.shipped); return `<tr class="ship"><th>${short[x.outcomes]}<br><span class="verdict small ${cls[m.verdict]}">${V[m.verdict]}</span><br><span class="small muted">${int(x.n_trials)} trials</span></th><td>${r.log_loss.toFixed(5)}</td><td>${m.log_loss.toFixed(5)}</td>
        <td>${diff(m.vs_reference)}</td><td>${m.vs_unadjusted ? diff(m.vs_unadjusted) : ""}</td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="small muted">The unadjusted rate is what the defense allowed in the first half, pooled toward the league, with no account of who it played. ${tc ? `The ${lvl}% interval on a report row held ${pct(tc.coverage, 1)} of the second half's differences as fitted${tc.coverage_with_drift !== null ? ` and ${pct(tc.coverage_with_drift, 1)} with the drift term` : ""} (${int(tc.n_cells)} rows).` : ""}${v.drift.available ? ` Drift was measured on ${seasonName(v.drift.season)}, never on the season being scored.` : ""}</p>
    ${units.length ? `<div class="splitname"><h3>Narrower than a team</h3><span class="tag">not in the report</span></div>
    <div class="scroll"><table class="tv"><thead><tr><th>Rows for</th><th>The team's row</th><th>The unit's own row</th><th>Difference (95% interval)</th><th>Only the rows a report would single out</th></tr></thead><tbody>
      ${units.map((x) => { const r = get(x, x.reference), m = get(x, x.shipped); return `<tr><th>${unitName[x.scope]}<br><span class="small">${short[x.outcomes].toLowerCase()}</span><br><span class="verdict small ${cls[m.verdict]}">${V[m.verdict]}</span><br><span class="small muted">${int(x.n_cells)} cells, ${int(x.n_trials)} trials</span></th><td>${r.log_loss.toFixed(5)}</td><td>${m.log_loss.toFixed(5)}</td>
        <td>${diff(m.vs_reference)}</td>
        <td>${x.picked && x.picked.vs_reference ? `${diff(x.picked.vs_reference)} <span class="verdict small ${cls[x.picked.verdict]}">${V[x.picked.verdict]}</span><br><span class="small muted">${int(x.picked.n_cells)} row${x.picked.n_cells === 1 ? "" : "s"}, ${int(x.picked.n_trials)} trials</span>` : `<span class="small muted">none cleared the bar</span>`}</td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="small muted">Here the reference is the team's own row, applied whoever is on the floor. A lineup's row, or a player's, has to beat that to be worth showing. A possession counts once for each player on the floor who has a row. The last column asks the same question of only the rows a report would single out: those at least ${units[0].picked ? pct(units[0].picked.min_probability) : "90%"} likely, after the first half, to sit more than a point from the team's other possessions.</p>` : ""}
    <p class="note plain">Log-loss interval: ${esc(v.bootstrap)}. A verdict needs that whole interval on one side of zero.</p>
  </div>`;
}

// ---------- view: hot hand ----------
function renderHotHand() {
  const h = S.hot_hand;
  const headHtml = `<div class="pagehead"><div><h2>Hot hand</h2>
    <p class="muted">After a player makes a shot, does he finish his team's next possession more often, and does that possession score more? One league-wide question, kept apart from the team reports.</p></div></div>`;
  if (!h) { $("#view").innerHTML = `${headHtml}<div class="sheet"><p class="note plain">This export has no hot-hand report. It was made before the report existed; run the export again to add it.</p></div>`; return; }
  if (!h.available) { $("#view").innerHTML = `${headHtml}<div class="sheet"><p class="note plain">Not available: ${esc(h.reason || "too little data")}.</p></div>`; return; }
  const lvl = Math.round(h.interval_level * 100), synth = h.data_kind === "synthetic";
  const mk = h.after.find((a) => a.prior === "make"), ms = h.after.find((a) => a.prior === "miss"), d = h.make_minus_miss;
  const p2 = (x) => `${signed(x * 100, 1)}`, f3 = (x) => signed(x, 3);
  const span = Math.max(0.001, ...[mk, ms].flatMap((a) => a.share_vs_expected.interval.map(Math.abs)), ...d.share.interval.map(Math.abs)) * 1.1;
  const spanP = Math.max(0.001, ...[mk, ms].flatMap((a) => a.points_vs_team.interval.map(Math.abs)), ...d.points.interval.map(Math.abs)) * 1.1;
  const bar = (e, sp, label) => { const X = (v) => (150 + (v / sp) * 142).toFixed(1), clear = e.interval[0] > 0 || e.interval[1] < 0, col = !clear ? "#6E7580" : e.value >= 0 ? "#2447D6" : "#C8253A";
    return `<svg class="ivl" viewBox="0 0 300 34" role="img" aria-label="${esc(label)}"><line class="axis" x1="4" y1="17" x2="296" y2="17"/><line class="zero" x1="150" y1="4" x2="150" y2="30"/>
      <line class="bar" x1="${X(e.interval[0])}" y1="17" x2="${X(e.interval[1])}" y2="17" stroke="${col}"/><circle class="dot" cx="${X(e.value)}" cy="17" r="5.5" stroke="${col}"/></svg>`; };
  const shareRow = (name, a) => `<tr><th>${name}</th><td>${pct(a.share, 1)}</td><td>${pct(a.expected_share, 1)}</td><td>${p2(a.share_vs_expected.value)} pts<br><span class="small muted">${p2(a.share_vs_expected.interval[0])} to ${p2(a.share_vs_expected.interval[1])}</span></td>
    <td class="plot">${bar(a.share_vs_expected, span, `${name}: ${p2(a.share_vs_expected.value)} points against expected`)}</td><td>${int(a.n)}</td></tr>`;
  const ptsRow = (name, a) => `<tr><th>${name}</th><td>${a.points.toFixed(3)}</td><td>${f3(a.points_vs_team.value)}<br><span class="small muted">${f3(a.points_vs_team.interval[0])} to ${f3(a.points_vs_team.interval[1])}</span></td>
    <td class="plot">${bar(a.points_vs_team, spanP, `${name}: ${f3(a.points_vs_team.value)} points against the team's average`)}</td><td>${int(a.n)}</td></tr>`;
  const VS = { rises: "He finishes the next one more often", falls: "He finishes the next one less often", no_clear_difference: "No clear difference in who finishes" };
  const VP = { more: "The possession scores more", fewer: "The possession scores less", no_clear_difference: "No clear difference in points" };
  const vc = (v) => (v === "no_clear_difference" ? "inconclusive" : v === "rises" || v === "more" ? "model_better" : "baseline_better");
  $("#view").innerHTML = `${headHtml}
  ${synth ? `<p class="note">These numbers come from simulated games, where nothing makes a player more likely to shoot after a make. They show the report runs; they say nothing about NBA players.</p>` : ""}
  <div class="sheet">
    <div class="splitname"><h3>Who finishes the next possession</h3><span class="verdict ${vc(h.verdict.share)}">${VS[h.verdict.share]}</span></div>
    <div class="scroll"><table class="tv hh"><thead><tr><th>The trip before</th><th>He finished</th><th>Expected</th><th>Difference (${lvl}% interval)</th><th class="plot"></th><th>Possessions</th></tr></thead><tbody>
      ${shareRow("After his make", mk)}${shareRow("After his miss", ms)}
      <tr class="ship"><th>Make minus miss</th><td></td><td></td><td>${p2(d.share.value)} pts<br><span class="small muted">${p2(d.share.interval[0])} to ${p2(d.share.interval[1])}</span></td><td class="plot">${bar(d.share, span, `Make minus miss: ${p2(d.share.value)} points`)}</td><td>${int(mk.n + ms.n)}</td></tr>
    </tbody></table></div>
    <div class="splitname"><h3>What the next possession scores</h3><span class="verdict ${vc(h.verdict.points)}">${VP[h.verdict.points]}</span></div>
    <div class="scroll"><table class="tv hh"><thead><tr><th>The trip before</th><th>Points</th><th>Against team average (${lvl}% interval)</th><th class="plot"></th><th>Possessions</th></tr></thead><tbody>
      ${ptsRow("After his make", mk)}${ptsRow("After his miss", ms)}
      <tr class="ship"><th>Make minus miss</th><td></td><td>${f3(d.points.value)}<br><span class="small muted">${f3(d.points.interval[0])} to ${f3(d.points.interval[1])}</span></td><td class="plot">${bar(d.points, spanP, `Make minus miss: ${f3(d.points.value)} points per possession`)}</td><td>${int(mk.n + ms.n)}</td></tr>
    </tbody></table></div>
    <p class="small muted" style="margin-top:14px">${esc(h.definition)} ${int(h.n_games)} games, season${h.seasons.length > 1 ? "s" : ""} ${h.seasons.map(seasonName).join(", ")}. Intervals: ${esc(h.bootstrap)}.</p>
    <p class="note plain">How to read it. The make-minus-miss row is the hot-hand number. Both groups are players who took their team's last shot, so comparing one with the other removes whatever is true of recent shooters in general. It is an association. Play-by-play does not say who a play was run for, so this cannot tell a team looking for the hot player from a player hunting his own shot, or from a night when the matchup favoured him from the start.</p>
  </div>`;
}

// ---------- routing and boot ----------
const RENDER = { method: renderMethod, team: renderTeam, replay: renderReplay, scorecard: renderScorecard, hothand: renderHotHand, ask: renderAsk };
let lastView = null;

async function render() {
  const view = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "method";
  document.querySelectorAll(".tabs a").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  document.title = `${{ method: "Method", team: "Team report", replay: "Replay", scorecard: "Scorecard", hothand: "Hot hand", ask: "Ask" }[view]} | Tendency Board`;
  const keep = view === lastView ? window.scrollY : 0;
  try { await RENDER[view](); } catch (e) { showError(e); }
  if (view !== lastView) { lastView = view; window.scrollTo(0, 0); } else window.scrollTo(0, keep);
}

async function boot() {
  const top = document.querySelector(".top");
  const setTop = () => document.querySelector("#tape").style.setProperty("top", `${top.offsetHeight}px`);
  setTop();
  window.addEventListener("resize", setTop);
  try {
    await Promise.all(SCHEMAS.map(async (n) => {
      const res = await fetch(`${DATA}schemas/${n}.schema.json`);
      if (!res.ok) throw new Error(`Could not load the ${n} schema (HTTP ${res.status}). The data folder is missing or incomplete; run the pipeline's export step.`);
      S.schemas[n] = await res.json();
    }));
    S.manifest = await getJSON("manifest.json", "manifest");
    const f = S.manifest.files;
    [S.teams, S.games, S.scorecard] = await Promise.all([getJSON(f.teams, "teams"), getJSON(f.games, "games"), getJSON(f.scorecard, "scorecard")]);
    for (const name of OPTIONAL) {                       // listed in the manifest from contract 1.2 on; absent in older exports
      if (!f[name]) continue;
      const res = await fetch(`${DATA}schemas/${name}.schema.json`);
      if (!res.ok) throw new Error(`Could not load the ${name} schema (HTTP ${res.status}). The data folder is incomplete; run the pipeline's export step.`);
      S.schemas[name] = await res.json();
      S[name] = await getJSON(f[name], name);
    }
    const askSchema = await fetch(`${DATA}schemas/ask_response.schema.json`);       // contract 1.8; an older export has none and Ask says so
    if (askSchema.ok) S.schemas.ask_response = await askSchema.json();
    for (const d of [S.teams, S.games, S.scorecard, S.tendency_validation, S.hot_hand]) if (d && d.data_kind !== S.manifest.data_kind) throw new ContractProblem("Files disagree about whether the data is synthetic or real.");
  } catch (e) { showError(e); return; }
  S.teams.byId = new Map(S.teams.teams.map((t) => [t.team_id, t]));
  S.teams.players = new Map(S.teams.teams.flatMap((t) => t.players.map((p) => [p.player_id, p.name])));
  if (S.manifest.data_kind === "synthetic") {
    const tape = $("#tape");
    tape.hidden = false;
    tape.innerHTML = `<strong>Synthetic sample data</strong> Placeholder teams and players. Every number on this page is simulated. Not NBA results.`;
  }
  $("#foot").textContent = `Data ${S.manifest.data_kind}, exported ${day(S.manifest.generated_at)}. Contract version ${S.manifest.contract_version}. Built from public play-by-play; not affiliated with the NBA.${S.manifest.data_kind === "real" ? " Statistics from NBA.com." : ""}`;
  window.addEventListener("hashchange", render);
  document.addEventListener("keydown", (e) => {
    if (lastView !== "replay" || e.altKey || e.metaKey || e.ctrlKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "ArrowRight") { const b = $("#next"); if (b && !b.disabled) b.click(); }
    if (e.key === "ArrowLeft") { const b = $("#prev"); if (b && !b.disabled) b.click(); }
  });
  await render();
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    try { await navigator.serviceWorker.register("sw.js"); } catch (e) { /* not available in this frame; the app works without it */ }
  }
}

boot();
