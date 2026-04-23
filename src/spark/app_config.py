from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SparkConfig:
    app_name: str
    master: str
    driver_memory: str
    shuffle_partitions: str
    log_level: str


@dataclass
class DataConfig:
    input_path: str
    clusters_csv_path: str
    profiles_csv_path: str
    centers_csv_path: str
    metrics_json_path: str
    predictions_csv_path: str
    model_root: str


@dataclass
class PreprocessingConfig:
    min_non_null_ratio: float
    target_n: int
    imputer_strategy: str


@dataclass
class TrainingConfig:
    k_min: int
    k_max: int
    seed: int
    metric_name: str
    distance_measure: str


@dataclass
class AppConfig:
    spark: SparkConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    base_dir: Path


def load_config(config_path: str | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return AppConfig(
        spark=SparkConfig(**data["spark"]),
        data=DataConfig(**data["data"]),
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        training=TrainingConfig(**data["training"]),
        base_dir=path.parent.resolve(),
    )