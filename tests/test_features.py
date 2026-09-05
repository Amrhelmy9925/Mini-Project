"""Taste-test the feature kitchen (features.py + data.py).

6yo version: kitchen turns recipe cards (dicts) into number boxes (matrix).
We check: missing ingredient doesn't crash, zero distance works, new PU_DO works.
"""

import pandas as pd
import pytest

from prodml.data import clean_data, split_data
from prodml.features import get_features, make_features


def _tiny_df():
    # Minimal raw dataframe that clean_data() accepts
    return pd.DataFrame(
        [
            {
                "VendorID": "1",
                "RatecodeID": "1",
                "store_and_fwd_flag": "N",
                "PULocationID": "100",
                "DOLocationID": "200",
                "payment_type": "1",
                "trip_distance": 2.0,
                "fare_amount": 10.0,
                "tpep_pickup_datetime": "2026-01-01 10:00:00",
                "tpep_dropoff_datetime": "2026-01-01 10:15:00",
                "total_amount": 13.0,
                "passenger_count": 1.0,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tip_amount": 2.0,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
            },
            {
                "VendorID": "2",
                "RatecodeID": "1",
                "store_and_fwd_flag": "N",
                "PULocationID": "101",
                "DOLocationID": "201",
                "payment_type": "2",
                "trip_distance": 5.0,
                "fare_amount": 20.0,
                "tpep_pickup_datetime": "2026-01-01 11:00:00",
                "tpep_dropoff_datetime": "2026-01-01 11:20:00",
                "total_amount": 25.0,
                "passenger_count": 2.0,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tip_amount": 3.0,
                "tolls_amount": 1.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
            },
        ]
    )


def test_clean_and_split():
    df = clean_data(_tiny_df())
    assert "duration" in df.columns
    assert (df["duration"] > 0).all()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0


def test_make_features_shape():
    df = clean_data(_tiny_df())
    X = df.drop("duration", axis=1)
    # split manually in half to avoid train_test_split randomness on 2 rows
    X_train, X_test = X.iloc[:1], X.iloc[1:]
    dv, X_train_dv, X_test_dv = make_features(X_train, X_test)
    assert X_train_dv.shape[0] == 1
    assert X_test_dv.shape[0] == 1
    assert X_train_dv.shape[1] == X_test_dv.shape[1]


@pytest.mark.parametrize(
    "case",
    [
        "missing_category",  # VendorID = None
        "zero_distance",  # trip_distance = 0
        "unseen_pudo",  # PU_DO never seen in train
    ],
)
def test_feature_edge_cases(case):
    df = clean_data(_tiny_df())
    X = df.drop("duration", axis=1)
    X_train, X_test = X.iloc[:1].copy(), X.iloc[1:].copy()

    if case == "missing_category":
        X_test.loc[:, "VendorID"] = None
    elif case == "zero_distance":
        X_test.loc[:, "trip_distance"] = 0.0
    elif case == "unseen_pudo":
        X_test.loc[:, "PULocationID"] = "999999"
        X_test.loc[:, "DOLocationID"] = "888888"

    # Must not crash; unseen categories become all-zero
    dv, X_train_dv, X_test_dv = make_features(X_train, X_test)
    assert X_test_dv.shape[0] == 1
    assert X_test_dv.shape[1] == X_train_dv.shape[1]
