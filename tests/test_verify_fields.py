"""verify-fields on a feed shaped like the live CDN one."""
def test_verify_fields_accepts_a_feed_with_no_area_when_shots_have_coordinates(tmp_path):
    """The live CDN feed (first fetched 2026-11-21) carries xLegacy / yLegacy but no `area`."""
    import json
    from tendency import cdn_fields as F
    from tendency.verify_fields import verify
    from conftest import AWAY, HOME, Game
    g = Game().start().shot(700, HOME, 101, made=True).shot(680, AWAY, 201, made=False, three=True).end()
    doc = g.pbp("0022600001")
    for a in doc[F.GAME][F.ACTIONS]:
        a.pop(F.AREA, None)
        a.update({F.X_LEGACY: 30.0, F.Y_LEGACY: 120.0})
    f = tmp_path / "pbp.json"
    f.write_text(json.dumps(doc))
    rows, _ = verify(f, None)
    by_item = {item: status for status, item, _ in rows}
    assert by_item["action key 'area'"] == "CHECK"
    assert by_item["no shot `area` in this file: every shot has legacy coordinates instead"] == "OK"
    for a in doc[F.GAME][F.ACTIONS]:
        a.pop(F.X_LEGACY)
    f.write_text(json.dumps(doc))
    rows, ok = verify(f, None)
    assert not ok and any(s == "MISSING" and "legacy coordinates" in item for s, item, _ in rows)
