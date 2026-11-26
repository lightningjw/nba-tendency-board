"""Every name we assume about the cdn.nba.com feeds lives in this one file.

STATUS: play-by-play names and values are VERIFIED against three full seasons (3,690
games) from the shufinskiy/nba_data mirror of the feed. Boxscore names are verified
against the nba_api documentation only. The schedule shape is still unverified.
ASSUMPTIONS.md has the detail. `python -m tendency verify-fields <pbp.json> <box.json>`
re-checks any single game.
"""
from __future__ import annotations

# --- URLs -------------------------------------------------------------------
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

# --- play-by-play: container --------------------------------------------------
GAME = "game"
ACTIONS = "actions"
GAME_ID = "gameId"

# --- play-by-play: per-action keys --------------------------------------------
ACTION_NUMBER = "actionNumber"
ORDER_NUMBER = "orderNumber"
CLOCK = "clock"                      # ISO-8601 duration, e.g. "PT11M42.00S"
PERIOD = "period"
TEAM_ID = "teamId"
PERSON_ID = "personId"
ACTION_TYPE = "actionType"
SUB_TYPE = "subType"
DESCRIPTOR = "descriptor"
DESCRIPTION = "description"
SHOT_RESULT = "shotResult"           # "Made" | "Missed"
SCORE_HOME = "scoreHome"             # string
SCORE_AWAY = "scoreAway"             # string
AREA = "area"
SHOT_DISTANCE = "shotDistance"       # feet
X_LEGACY = "xLegacy"                 # tenths of a foot, lateral, hoop at 0
Y_LEGACY = "yLegacy"                 # tenths of a foot, toward midcourt, hoop at 0
FOUL_DRAWN_PERSON_ID = "foulDrawnPersonId"
ASSIST_PERSON_ID = "assistPersonId"
STEAL_PERSON_ID = "stealPersonId"
BLOCK_PERSON_ID = "blockPersonId"
QUALIFIERS = "qualifiers"            # list of tags: "inpenalty", "startperiod", "deadball", "team", ...
POSSESSION = "possession"            # teamId with the ball; used only as a cross-check

# --- play-by-play: actionType values (compared lower-cased) -------------------
T_PERIOD = "period"
T_2PT = "2pt"
T_3PT = "3pt"
T_FREETHROW = "freethrow"
T_REBOUND = "rebound"
T_TURNOVER = "turnover"
T_FOUL = "foul"
T_SUBSTITUTION = "substitution"
T_TIMEOUT = "timeout"
T_JUMPBALL = "jumpball"
T_HEAVE = "heave"                    # new in 2025-26: a missed end-of-period heave, charged to the team, no shooter
ON_FLOOR_ACTION_TYPES = {T_2PT, T_3PT, T_FREETHROW, T_REBOUND, T_TURNOVER, T_FOUL, "steal", "block", T_JUMPBALL, "violation"}
KNOWN_ACTION_TYPES = {
    T_PERIOD, T_2PT, T_3PT, T_FREETHROW, T_REBOUND, T_TURNOVER, T_FOUL,
    T_SUBSTITUTION, T_TIMEOUT, T_JUMPBALL, T_HEAVE,
    "steal", "block", "violation", "stoppage", "instantreplay", "ejection", "game",
}

# --- play-by-play: subType values (compared lower-cased) ----------------------
ST_PERIOD_START = "start"
ST_PERIOD_END = "end"
ST_REBOUND_OFFENSIVE = "offensive"
ST_REBOUND_DEFENSIVE = "defensive"
ST_FOUL_OFFENSIVE = "offensive"
ST_SUB_IN = "in"
ST_SUB_OUT = "out"
TECHNICAL_MARKER = "technical"       # searched in subType, descriptor and description
FT_TRIP_PATTERN = r"(\d)\s*of\s*(\d)"  # freethrow subType, e.g. "1 of 2"
SHOT_MADE = "made"
FOUL_DOUBLE = "double"               # foul descriptor; double fouls do not add to team fouls
# freethrow descriptors after which the shooting team keeps the ball
FT_RETAIN_DESCRIPTORS = {"flagrant", "clear-path", "transition take", "away-from-play"}
FT_AWAY_FROM_PLAY = "away-from-play"   # keeps the ball only as a lone "1 of 1" with no made basket on the play
FT_KEEP_BALL_DESCRIPTORS = FT_RETAIN_DESCRIPTORS - {FT_AWAY_FROM_PLAY}   # always keep the ball, even after a made basket
# The live CDN feed (fetched with nba_api, 2026-27) leaves technical and away-from-play free throws unmarked;
# the season archives mark them. The foul that awarded the free throw always says what it was:
FT_TECHNICAL = "technical"
FOUL_DESCRIPTOR_TO_FT = {            # foul descriptor -> the free throw descriptor the archives would carry
    "away-from-play": "away-from-play",
    "flagrant-type-1": "flagrant",
    "flagrant-type-2": "flagrant",
    "clear-path": "clear-path",
    "transition take": "transition take",
}

# --- play-by-play: shot `area` values -> shot family --------------------------
AREA_TO_FAMILY = {
    "restricted area": "rim",
    "in the paint (non-ra)": "paint",
    "mid-range": "midrange",
    "left corner 3": "corner3",
    "right corner 3": "corner3",
    "above the break 3": "above_break3",
    "backcourt": "above_break3",     # heave rule handles the end-of-period ones
}

# --- play-by-play: shot `descriptor` and `subType` -> action style -----------------
# Profiled on 657,380 field-goal attempts, 2023-24 to 2025-26 (31 descriptor values, 4 subTypes), and on
# 42,347 from the 2026-27 live feed (same values). A style describes the SHOT the scorer saw. It is not a
# play call: "cut_lob" says the shot was a cutting layup or an alley-oop, not what set was run to get it.
# Rules are tried in order and the first that matches wins; a rule matches when the lower-cased descriptor
# contains any of its words, or (for the rules with no words) on the subType alone. Every attempt gets one.
#   style              share   what falls in it
#   plain_jumper       29.2%   Jump Shot with no descriptor (or only "bank"): 94% threes, 97% of makes assisted
#   drive              14.7%   driving layup, dunk, hook, finger roll, reverse, bank
#   pullup             13.3%   pullup, running pullup
#   running             8.8%   running layup, dunk, jump shot, finger roll, reverse
#   floater             8.7%   floating, driving floating
#   turnaround_hook     6.8%   any hook that is not driving; turnaround and fadeaway jump shots
#   cut_lob             5.7%   cutting, alley-oop, running alley-oop
#   step_back           5.3%   step back
#   putback             4.5%   tip, putback
#   plain_finish        3.1%   layup or dunk with no movement word (none, reverse, finger roll)
SHOT_SUBTYPE_JUMP = "jump shot"
SHOT_SUBTYPE_HOOK = "hook"
ACTION_STYLE_RULES = [               # (style, descriptor words); order matters
    ("putback", ("tip", "putback")),
    ("cut_lob", ("cutting", "alley-oop")),
    ("floater", ("floating",)),
    ("drive", ("driving",)),
    ("pullup", ("pullup",)),
    ("step_back", ("step back",)),
    ("turnaround_hook", ("turnaround", "fadeaway")),
]
ACTION_STYLE_RUNNING = "running"     # after the rules above: "running" with no pullup / alley-oop
ACTION_STYLES = ["plain_jumper", "drive", "pullup", "running", "floater", "turnaround_hook", "cut_lob", "step_back", "putback", "plain_finish"]
ACTION_STYLE_LABELS = ["Jump shot, no move noted", "Drive", "Pull-up", "Running", "Floater", "Turnaround or hook", "Cut or lob",
                       "Step-back", "Putback or tip", "Layup or dunk, no move noted"]

# --- play-by-play: `qualifiers` the scorer adds to shots and free throws ------------------
Q_FAST_BREAK = "fastbreak"
Q_SECOND_CHANCE = "2ndchance"
Q_FROM_TURNOVER = "fromturnover"

# --- boxscore -----------------------------------------------------------------
BOX_GAME_TIME_UTC = "gameTimeUTC"
BOX_HOME = "homeTeam"
BOX_AWAY = "awayTeam"
BOX_TEAM_ID = "teamId"
BOX_TEAM_TRICODE = "teamTricode"
BOX_TEAM_NAME = "teamName"
BOX_TEAM_CITY = "teamCity"
BOX_PLAYERS = "players"
BOX_PLAYER_ID = "personId"
BOX_PLAYER_NAME = "name"
BOX_PLAYER_STARTER = "starter"       # "1" for the five starters

# --- schedule -----------------------------------------------------------------
SCHED_ROOT = "leagueSchedule"
SCHED_GAME_DATES = "gameDates"
SCHED_GAMES = "games"
SCHED_GAME_ID = "gameId"
SCHED_GAME_TIME_UTC = "gameDateTimeUTC"
SCHED_GAME_STATUS = "gameStatus"     # 3 == final
SCHED_STATUS_FINAL = 3

# --- rules of thumb that are assumptions, not feed fields ----------------------
REGULATION_PERIOD_SECONDS = 720.0
OVERTIME_PERIOD_SECONDS = 300.0
TIMEOUTS_PER_GAME = 7                # ignores the Q4 cap and overtime grants
PENALTY_TEAM_FOULS = 4               # in the penalty once the defense has this many in a quarter
PENALTY_TEAM_FOULS_OT = 3            # ... or this many in an overtime period; plus the last-two-minutes rule
HEAVE_MAX_CLOCK_S = 3.0
# Shot location from legacy coordinates (tenths of a foot, hoop at 0,0), used when `area` is absent (the
# live CDN feed fetched 2026-11-21 has none) and tuned against `area` on 160k shots from 2023-26:
RIM_MAX_FT = 4.4                     # the feed's "Restricted Area" runs to about 4.4 ft, not 4.0
CORNER3_MAX_Y = 89.5                 # corner threes end where the line starts to arc
PAINT_HALF_WIDTH = 80.0              # lane is 16 ft wide
PAINT_MAX_Y = 137.5                  # free-throw line, measured from the hoop
HEAVE_MIN_DISTANCE_FT = 40.0
# Tempo: seconds from the start of the possession to its FIRST shot, turnover or trip to the line. The feed
# has no shot clock, so this is the period clock, and after a made basket it includes the inbound.
SHOT_CLOCK_S = 24.0                  # tempo is only defined when the period had at least this much left at the start
TEMPO_EARLY_MAX_S = 8.0              # "early": the first action came within this many seconds
TEMPO_LATE_MIN_S = 16.0              # "late": it came at or after this many
TEMPO_CLASSES = ["early", "middle", "late"]
# The live CDN feed stamps a defensive rebound with the missed shot's clock (7,130 of 7,130 in 120 games of
# 2026-27); the season archives stamp it when the ball was secured (median 3 s later, same clock in 6%). A
# possession that starts off such a rebound would look that much slower. When a game's rebounds are untimed,
# the tempo labels (and only they) start the possession this many seconds after the rebound's clock.
UNTIMED_REBOUND_LAG_S = 3.0
UNTIMED_REBOUND_MIN_SHARE = 0.9      # a game counts as untimed when this share of its defensive rebounds match the shot's clock
UNTIMED_REBOUND_MIN_N = 10
# Two-for-one: getting the ball with this much left in a period means a quick shot buys a second possession.
TWO_FOR_ONE_MIN_START_S = 30.0       # window is (min, max] seconds left when the possession starts
TWO_FOR_ONE_MAX_START_S = 45.0
TWO_FOR_ONE_SHOOT_BY_S = 27.0        # "took it": the first action came with at least this much left
SYNTHETIC_MARKER = "_synthetic"      # top-level key our generator writes; never in real data


def season_from_game_id(game_id: str) -> int:
    """'0022500061' -> 2025. Assumes characters 3-4 are the season start year."""
    yy = int(str(game_id)[3:5])
    return 2000 + yy
