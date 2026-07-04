from pathlib import Path
from adaptivedose.config import load_data_config, DataConfig

def test_load_data_config_reads_yaml(tmp_path):
    cfg_text = """
cohort:
  required_tracks: ["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"]
  min_asa: 1
  max_asa: 4
  ane_type: "General"
tracks:
  bis: "BIS/BIS"
  propofol_rate: "Orchestra/PPF20_RATE"
  remifentanil_rate: "Orchestra/RFTN20_RATE"
  map: "Solar8000/ART_MBP"
  hr: "Solar8000/HR"
  spo2: "Solar8000/PLETH_SPO2"
  etco2: "Solar8000/ETCO2"
resample:
  interval_sec: 10
split:
  test_frac: 0.15
  val_frac: 0.15
  seed: 42
cache_dir: ".vitaldb_cache"
output_dir: "data/processed"
"""
    p = tmp_path / "data.yaml"
    p.write_text(cfg_text)
    cfg = load_data_config(p)
    assert isinstance(cfg, DataConfig)
    assert cfg.resample.interval_sec == 10
    assert cfg.split.seed == 42
    assert "BIS/BIS" in cfg.cohort.required_tracks
    assert cfg.tracks.map == "Solar8000/ART_MBP"

def test_load_data_config_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_data_config(tmp_path / "nope.yaml")
