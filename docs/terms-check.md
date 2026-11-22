# Terms check (roadmap step 1.0a)

Read on 2026-11-21. Research only: nothing published was changed because of it. This is a reading of public
terms by a non-lawyer, written for one hobby project; it is not legal advice.

## Sources

| Source | What it is | Read |
|---|---|---|
| [NBA.com Terms of Use](https://www.nba.com/termsofuse) | Governs nba.com, its apps and feeds. Last updated July 13, 2026 | Sections 1 and 9 |
| [NBA.com/Stats FAQ](https://www.nba.com/stats/help/faq) | The stats site's own answer on downloads | Download question |
| [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) | The mirror this repo loads seasons from | README, LICENSE (Apache-2.0) |
| [swar/nba_api](https://github.com/swar/nba_api) | Client for stats.nba.com and the CDN feed | README, LICENSE (MIT) |
| [anovick1/HoopsOracle](https://github.com/anovick1/HoopsOracle), [dblackrun/pbpstats](https://github.com/dblackrun/pbpstats) | Comparable public projects | README, LICENSE |

## What the terms say

**Automated access.** Section 1 of the Terms of Use has no clause on robots, scraping or data mining. It allows
viewing, and downloading to one computer for personal, non-commercial use, and it bars reproducing or
republishing the content without written permission. The stats FAQ is blunter: the statistics are not offered for
download, for academic or personal use. In practice stats.nba.com does not answer cloud hosts and the CDN feed
refuses plain requests; `nba_api` gets through by sending browser-like headers. So: no explicit ban on automated
access was found, no permission for it either, and the site signals by its behaviour that it does not want it.

**Republishing statistics (Section 9, "NBA Statistics").** Using the statistics means accepting seven conditions:

1. Prominent attribution to NBA.com on any use, display or publication.
2. Only for legitimate news reporting or for private, non-commercial purposes.
3. No sponsorship or commercial identification.
4. No gambling use, legal gambling included.
5. No fantasy game or other commercial product or service.
6. No product or service that presents a "live, near-live, or other real-time or archived play-by-play account"
   of a game. Archived is named; no exception is stated.
7. No site, product or service featuring a database of comprehensive, regularly updated statistics, without
   prior consent.

There is no volume limit and no clause on training models.

**The mirror.** `shufinskiy/nba_data` is Apache-2.0. Its README says the data was collected from stats.nba.com,
data.nba.com and pbpstats.com with the author's scripts. The licence covers what the author can license (his
code and his packaging). It cannot grant rights over the underlying NBA statistics, and the README does not
claim to.

**nba_api.** MIT for the code. Its README says data retrieved with it comes from NBA.com and is subject to the
NBA.com Terms of Use, and that the project is not affiliated with the NBA.

## Conservative reading for this project

| Thing | Reading | Why |
|---|---|---|
| Public repo with code only, plus instructions to download data yourself | Permitted | The code is yours. This is what nba_api, pbpstats and HoopsOracle do |
| Public repo containing per-possession rows, raw play-by-play, or a real export's replay files | Prohibited | Condition 6 (archived play-by-play account) and the Section 1 ban on republishing |
| Public site: model scorecard (log-loss, calibration) on real games | Unclear, low risk | Numbers about a model, not NBA statistics. Non-commercial. Attribute the source anyway |
| Public site: team tendency tables with real team and player names | Unclear | Derived, shrunk aggregates with attribution and no commercial use fit the spirit of condition 2, but a public site is not "private", and nightly updates across 30 teams move toward condition 7 |
| Public site: replay view of real games (possession by possession) | Prohibited on a conservative reading | It is an archived play-by-play depiction: condition 6 |
| Public site: live mode on real games (Phase 3) | Prohibited on a conservative reading | Condition 6 names live and near-live |
| Per-game pregame files with real names in the public repo | Unclear, low risk | Predictions and baselines, no play-by-play. `SHA256SUMS` alone avoids the question |
| Running `scripts/fetch_nba_api.py` at home, a few requests a day | Unclear | Not expressly banned; not licensed; the FAQ says no downloads |

## How comparable projects handle it

- **nba_api**: ships code only, points users at the NBA.com terms, disclaims affiliation.
- **pbpstats** (the package): code only, MIT; the user fetches data.
- **HoopsOracle**: code only; the README says play-by-play comes from the live CDN and gives commands to
  download seasons yourself. No licence file, no data in the repo.
- **shufinskiy/nba_data**: the exception. It republishes the data itself, under Apache-2.0, and has done so
  publicly for years. That shows what is tolerated, not what is permitted.

## Recommendation

Publish code and derived aggregates; never publish play-by-play.

- Keep the public repo as it is: code, tests, schemas, the synthetic sample. No `data/`, no real export.
- On the public site, real data may back the **scorecard** and the **team reports**, each with a visible
  "Statistics from NBA.com" attribution, no ads, no sponsorship. Keep the **replay view on synthetic games**
  there; run real replays locally and show them in person.
- For the prospective log, commit `SHA256SUMS` only (`PUBLISH=sums`). It timestamps the log without publishing
  anything. The per-game files stay local and can be shown on request.
- Treat Phase 3 **live mode on real games as local-only**, or build the roadmap's "replay as live" fallback.
- If this ever becomes commercial, or gets traffic, ask NBA.com for consent before anything else.
