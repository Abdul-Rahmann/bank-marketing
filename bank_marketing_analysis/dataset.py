import pandas as pd
import zipfile
import urllib.request
from loguru import logger
from pathlib import Path

from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

from .config import DataConfig

def download_data(output_dir: Path):
    """Down and extract data if missing"""
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "bank.zip"

    logger.info(f"Downloading dataset from {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    logger.info("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    zip_path.unlink()
    if (output_dir / "bank.csv").exists():
        (output_dir / "bank.csv").unlink()

    logger.success(f"Data is ready in {output_dir}")

def load_raw_data(config: DataConfig = None) -> pd.DataFrame:
    """Load raw bank marketing data"""
    raw_path = Path(config.raw_path)

    if not raw_path.exists():
        logger.warning(f"File not found at {raw_path}. Initializing download...")
        download_data(raw_path.parent)

    logger.info("Reading raw data: {raw_path.name}")
    return pd.read_csv(raw_path, sep=";")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: target encoding and label encoding for features"""
    df = df.copy()

    df['target'] = df['y'].map({'yes':1,'no':0})
    df = df.drop('y', axis=1)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logger.debug(f"Preprocessed DataFrame: {df.shape}")

    return df


def get_train_test_data(config: DataConfig = None):
    """The main entry points: return X_train, X_test, y_train, y_test and feature names"""
    if config is None:
        config = DataConfig()

    raw_df = load_raw_data(config)
    processed_df = preprocess_data(raw_df)

    X = processed_df.drop('target', axis=1)
    y = processed_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    logger.info(f"Data Split Complete: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Target Baseline (Positive Rate): {y.mean():.2%}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    from .config import DataConfig
    logger.info("Starting Data Pipeline Test...")

    cfg = DataConfig()

    try:
        X_train, X_test, y_train, y_test, features = get_train_test_data(cfg)

        print("\n--- Pipeline Test Results ---")
        print(f"Feature Count: {len(features)}")
        print(f"X_train Shape: {X_train.shape}")
        print(f"y_train Mean (Class Balance): {y_train.mean():.4f}")
        print(f"First 5 features: {features[:5]}")

        if X_train.isnull().values.any():
            logger.warning(f"Found NaN values in the processed data!")
        else:
            logger.success("No missing values found. Ready for training")

    except Exception as e:
        logger.error(f"Pipeline failed with error {e}")
