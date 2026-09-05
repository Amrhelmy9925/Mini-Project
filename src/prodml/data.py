"""Data loading, cleaning, and train/test split.

Notebook-style flow: `load_data()` returns a raw DataFrame,
`clean_data(df)` takes it and returns the cleaned one,
`split_data(df)` returns the train/test arrays. Each step's output
feeds the next step's input — same as notebook cells run in order.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings


def load_data() -> pd.DataFrame:
    """Read the raw parquet into a DataFrame.

    Path comes from settings (env var DATA_PATH overrides), so no
    hardcoded machine-specific path.
    """
    return pd.read_parquet(settings.data_path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Build target, drop leaky columns, stringify categoricals, filter
    outliers. Takes the raw df, returns a new cleaned df (input untouched).
    """
    df = data.copy()

    # Ensure datetime dtype (to_datetime is idempotent if already datetime).
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Target: trip duration in minutes. Subtracting datetimes gives a
    # timedelta; .dt.total_seconds() makes it a float of seconds; /60 = minutes.
    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Drop target leakage (total_amount includes fare, which encodes
    # duration) and the raw datetimes already used to build the target.
    to_drop = [
        "total_amount",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df = df.drop(columns=to_drop)

    # Columns DictVectorizer must one-hot encode.
    categorical_cols = [
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    ]
    # Stringify so DictVectorizer one-hots each value instead of treating
    # the column as one numeric feature. fillna first so NaN becomes a
    # stable category instead of breaking astype(str) on nullable dtypes.
    df[categorical_cols] = df[categorical_cols].fillna("nan").astype(str)

    # Sanity filters: positive distance/fare, duration 1..180 min.
    # One combined mask: single pass instead of three DataFrame copies.
    df = df[
        df["trip_distance"].gt(0)
        & df["fare_amount"].gt(0)
        & df["duration"].between(1, 180)
    ]

    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Split the cleaned df into X_train/X_test/y_train/y_test."""
    # Remaining numeric columns: fill stray NaN with 0.
    X = df.drop("duration", axis=1).fillna(0)
    y = df["duration"]

    # test_size / random_state come from config so env-overridable.
    return train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )


def get_training_data() -> tuple:
    """One-call convenience: load -> clean -> split. Same as running all
    three cells in order in a notebook."""
    return split_data(clean_data(load_data()))
