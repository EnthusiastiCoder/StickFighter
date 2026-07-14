from stickfighter.game import save


def test_load_returns_defaults_when_no_file_exists(tmp_path):
    data = save.load(tmp_path / "save.json")
    assert data == save.SaveData(total_coins=0, gore_enabled=True)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "save.json"
    data = save.SaveData(total_coins=42, gore_enabled=False)
    save.save(data, path)
    assert save.load(path) == data


def test_load_falls_back_to_defaults_on_corrupted_json(tmp_path):
    path = tmp_path / "save.json"
    path.write_text("not valid json {{{")
    assert save.load(path) == save.SaveData()


def test_load_falls_back_to_defaults_on_wrong_shaped_json(tmp_path):
    path = tmp_path / "save.json"
    path.write_text('{"total_coins": "not-a-number"}')
    assert save.load(path) == save.SaveData()


def test_load_tolerates_missing_fields(tmp_path):
    path = tmp_path / "save.json"
    path.write_text('{"total_coins": 7}')
    data = save.load(path)
    assert data.total_coins == 7
    assert data.gore_enabled is True  # default filled in
