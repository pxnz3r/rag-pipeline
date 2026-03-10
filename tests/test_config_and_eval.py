from pathlib import Path

from rag_pipeline.config import load_settings
from rag_pipeline.evaluation import generate_test_dataset


def test_load_settings_creates_directories(tmp_path: Path):
    settings = load_settings(base_dir=tmp_path)
    assert settings.data_dir.exists()
    assert settings.processed_dir.exists()
    assert settings.db_dir.exists()
    assert settings.lightrag_dir.exists()


def test_generate_test_dataset_shape():
    data = generate_test_dataset()
    assert len(data["questions"]) == 30
    assert len(data["ground_truths"]) == 30
