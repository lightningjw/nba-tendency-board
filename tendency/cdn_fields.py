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
HEAVE_MIN_DISTANCE_FT = 40.0
SYNTHETIC_MARKER = "_synthetic"      # top-level key our generator writes; never in real data


def season_from_game_id(game_id: str) -> int:
    """'0022500061' -> 2025. Assumes characters 3-4 are the season start year."""
    yy = int(str(game_id)[3:5])
    return 2000 + yy
