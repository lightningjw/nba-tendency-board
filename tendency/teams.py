"""The 30 NBA franchises, keyed by the feed's team id.

Built from data, not from memory: ids, cities and nicknames are the `team_id`, `team_city` and
`team_name` columns of the shufinskiy/nba_data matchups archives for 2023-24 to 2025-26, and every id
was matched against the team ids in the 3,690 play-by-play games on disk (30 of 30, same tricodes).

The id is the key. A game's tricode always comes from that game's own feed, so a rebrand or a
relocation shows the tricode of the day while city and nickname follow this table until it is edited.
An id that is not here falls back to the tricode, never to a guess.
"""
from __future__ import annotations

# team id: (tricode, city, nickname)
TEAMS: dict[int, tuple[str, str, str]] = {
    1610612737: ("ATL", "Atlanta", "Hawks"),
    1610612738: ("BOS", "Boston", "Celtics"),
    1610612739: ("CLE", "Cleveland", "Cavaliers"),
    1610612740: ("NOP", "New Orleans", "Pelicans"),
    1610612741: ("CHI", "Chicago", "Bulls"),
    1610612742: ("DAL", "Dallas", "Mavericks"),
    1610612743: ("DEN", "Denver", "Nuggets"),
    1610612744: ("GSW", "Golden State", "Warriors"),
    1610612745: ("HOU", "Houston", "Rockets"),
    1610612746: ("LAC", "LA", "Clippers"),
    1610612747: ("LAL", "Los Angeles", "Lakers"),
    1610612748: ("MIA", "Miami", "Heat"),
    1610612749: ("MIL", "Milwaukee", "Bucks"),
    1610612750: ("MIN", "Minnesota", "Timberwolves"),
    1610612751: ("BKN", "Brooklyn", "Nets"),
    1610612752: ("NYK", "New York", "Knicks"),
    1610612753: ("ORL", "Orlando", "Magic"),
    1610612754: ("IND", "Indiana", "Pacers"),
    1610612755: ("PHI", "Philadelphia", "76ers"),
    1610612756: ("PHX", "Phoenix", "Suns"),
    1610612757: ("POR", "Portland", "Trail Blazers"),
    1610612758: ("SAC", "Sacramento", "Kings"),
    1610612759: ("SAS", "San Antonio", "Spurs"),
    1610612760: ("OKC", "Oklahoma City", "Thunder"),
    1610612761: ("TOR", "Toronto", "Raptors"),
    1610612762: ("UTA", "Utah", "Jazz"),
    1610612763: ("MEM", "Memphis", "Grizzlies"),
    1610612764: ("WAS", "Washington", "Wizards"),
    1610612765: ("DET", "Detroit", "Pistons"),
    1610612766: ("CHA", "Charlotte", "Hornets"),
}


def city_and_nickname(team_id: int, tricode: str) -> tuple[str, str]:
    """(city, nickname) for a team id; ("", tricode) when the id is unknown, so the name degrades to the tricode."""
    t = TEAMS.get(int(team_id))
    return (t[1], t[2]) if t else ("", str(tricode))
