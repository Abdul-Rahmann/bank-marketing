from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

@dataclass
class DataConfig:
    raw_path: str = str(PROJ_ROOT / "data/raw/bank-full.csv")
    processed_dir:  str = str(PROJ_ROOT / "data/processed")
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    type: str = "random_forest"
    n_estimators: int = 100
    max_depth: Optional[int] = 10

@dataclass
class ExperimentConfig:
    run_name: str = "baseline"
    model_name: str = "bank_mode_dev"
    data: DataConfig = None
    model: ModelConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()

def load_config(config_path: str = "configs/train.yaml") -> ExperimentConfig:
    full_path = PROJ_ROOT / config_path

    if not full_path.exists():
        logger.warning(f"Config {config_path} not found. Using defaults")
        return ExperimentConfig()

    with open(full_path) as f:
        config_dict = yaml.safe_load(f) or {}

    data_config = DataConfig(**config_dict.get('data',{}))
    model_config = ModelConfig(**config_dict.get('model',{}))

    return ExperimentConfig(
        run_name=config_dict.get('run_name','baseline'),
        model_name=config_dict.get('model_name','bank_model_dev'),
        data=data_config,
        model=model_config
    )


if __name__ == "__main__":
    # Test 1: Load with defaults (no file)
    logger.info("Test 1: Loading default config...")
    cfg_default = load_config("non_existent.yaml")
    print(f"Default Run Name: {cfg_default.run_name}")
    print(f"Default Raw Path: {cfg_default.data.raw_path}")

    # Test 2: Create a dummy YAML to test loading
    import os

    test_yaml = PROJ_ROOT / "configs/test_config.yaml"
    test_yaml.parent.mkdir(exist_ok=True)

    with open(test_yaml, "w") as f:
        yaml.dump({
            "run_name": "test-experiment",
            "model": {"n_estimators": 500, "max_depth": 20}
        }, f)

    logger.info(f"Test 2: Loading from {test_yaml.name}...")
    cfg_custom = load_config("configs/test_config.yaml")
    print(f"Custom Run Name: {cfg_custom.run_name}")
    print(f"Custom Estimators: {cfg_custom.model.n_estimators}")

    # Cleanup test file
    test_yaml.unlink()
    logger.success("Config tests passed!")

