import numpy as np
import pandas as pd

from rul_service.config import ALL_COLUMNS, Config
from rul_service.data import Preprocessor, add_training_rul


def _toy_df(n_engines=2, cycles=40):
    rows = []
    rng = np.random.default_rng(0)
    for eid in range(1, n_engines + 1):
        for c in range(1, cycles + 1):
            row = {col: 0.0 for col in ALL_COLUMNS}
            row["engine_id"] = eid
            row["cycle"] = c
            row["op1"] = rng.normal(0, 1)
            row["op2"] = rng.normal(5, 2)
            row["s2"] = c + rng.normal(0, 0.5)  # has variance
            row["s3"] = 100 + c * 0.5
            row["s1"] = 1.0  # constant -> dropped
            rows.append(row)
    return pd.DataFrame(rows)[ALL_COLUMNS]


def test_rul_clipping():
    df = add_training_rul(_toy_df(), max_rul=10)
    assert df["RUL"].max() == 10
    assert df["RUL"].min() == 0


def test_constant_columns_dropped():
    df = _toy_df()
    pre = Preprocessor.fit(df, Config(constant_std_threshold=0.1))
    assert "s1" not in pre.feature_columns  # constant
    assert "op3" not in pre.feature_columns  # always dropped
    assert "s2" in pre.feature_columns


def test_scaling_range():
    df = _toy_df()
    pre = Preprocessor.fit(df)
    scaled = pre.select_and_scale(df)
    assert scaled.shape[1] == pre.n_features
    assert scaled.min() >= -1e-6 and scaled.max() <= 1 + 1e-6


def test_window_shapes_and_padding():
    df = add_training_rul(_toy_df(cycles=40), max_rul=125)
    pre = Preprocessor.fit(df)
    X, y = pre.make_train_sequences(df)
    assert X.shape[1] == pre.sequence_length
    assert X.shape[2] == pre.n_features
    assert len(X) == len(y)

    # Front-padding for short histories.
    short = pre.scale_array(df[pre.feature_columns].to_numpy()[:5])
    w = pre.window_from_scaled(short)
    assert w.shape == (pre.sequence_length, pre.n_features)
    assert np.allclose(w[0], 0.0)  # padded row


def test_preprocessor_roundtrip():
    pre = Preprocessor.fit(_toy_df())
    pre2 = Preprocessor.from_dict(pre.to_dict())
    assert pre2.feature_columns == pre.feature_columns
    assert pre2.data_min == pre.data_min
