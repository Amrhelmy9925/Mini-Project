"""Feature engineering: turn dataframe rows into sparse feature vectors.

Notebook-style flow: takes the split data from data.py, returns the
vectorizer and the vectorized matrices.
"""

from sklearn.feature_extraction import DictVectorizer

from .data import get_training_data


def make_features(X_train, X_test):
    """Vectorize train/test with a fresh DictVectorizer.

    Returns (dv, X_train_dv, X_test_dv).

    DictVectorizer turns a list of dicts into a sparse matrix:
    - numeric values become single columns
    - string values are one-hot encoded (one column per category)

    to_dict(orient="records") converts the DataFrame into a list of dicts,
    one dict per row: [{"VendorID": "1", "trip_distance": 1.5, ...}, ...].
    This is the input format DictVectorizer expects.

    fit_transform on train only: learns the vocabulary (which columns and
    which category values exist). transform on test reuses that vocabulary —
    unseen categories become all-zero instead of crashing, which is what
    keeps inference safe on new data.
    """
    dv = DictVectorizer(sparse=True)
    X_train_dv = dv.fit_transform(X_train.to_dict(orient="records"))
    X_test_dv = dv.transform(X_test.to_dict(orient="records"))
    return dv, X_train_dv, X_test_dv


def get_features():
    """One-call convenience: load/clean/split data, then vectorize.
    Returns everything train.py needs."""
    X_train, X_test, y_train, y_test = get_training_data()
    dv, X_train_dv, X_test_dv = make_features(X_train, X_test)
    return dv, X_train_dv, X_test_dv, y_train, y_test
