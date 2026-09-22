import copy

import numpy as np
import pandas as pd

from evidently.core.datasets import DataDefinition
from evidently.core.datasets import Dataset
from evidently.core.datasets import PandasDataset
from evidently.core.report import Report
from evidently.legacy.core import ColumnType
from evidently.presets import DataDriftPreset


def _make_frames():
    rng = np.random.default_rng(0)
    n = 100
    ref = pd.DataFrame({"feat": rng.normal(0, 1, n), "target": rng.integers(0, 2, n)})
    cur = pd.DataFrame(
        {
            "feat": rng.normal(0.5, 1, n),
            "target": rng.integers(0, 2, n),
            "execution_timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        }
    )
    return ref, cur


def test_strict_does_not_expand_definition():
    # evidentlyai/evidently#1504: with strict=True a user-provided definition is
    # used verbatim — unlisted columns are not auto-detected into it.
    _, cur = _make_frames()
    definition = DataDefinition(numerical_columns=["feat"], categorical_columns=["target"])
    snapshot = copy.deepcopy(definition)

    ds = PandasDataset(cur, data_definition=definition, strict=True)

    assert definition == snapshot  # caller's object untouched
    assert ds.data_definition.timestamp is None
    assert ds.data_definition.datetime_columns is None
    assert ds.data_definition.get_column_type("execution_timestamp") == ColumnType.Unknown
    assert ds.data_definition.get_column_type("feat") == ColumnType.Numerical
    assert ds.data_definition.get_column_type("target") == ColumnType.Categorical


def test_strict_shared_definition_mismatched_columns_report_runs():
    # The issue's crash scenario: same definition shared between reference and
    # current datasets where current has an extra metadata column. Without
    # strict the report raises
    # `ValueError: Column (execution_timestamp) is partially present in data`.
    ref, cur = _make_frames()
    definition = DataDefinition(numerical_columns=["feat"], categorical_columns=["target"])

    ds_ref = PandasDataset(ref, data_definition=definition, strict=True)
    ds_cur = PandasDataset(cur, data_definition=definition, strict=True)

    report = Report([DataDriftPreset()])
    report.run(current_data=ds_cur, reference_data=ds_ref)  # must not raise


def test_default_behavior_still_expands_definition():
    # Default (strict=False) preserves the historical auto-expansion behavior.
    _, cur = _make_frames()
    definition = DataDefinition(numerical_columns=["feat"], categorical_columns=["target"])

    ds = PandasDataset(cur, data_definition=definition)

    assert definition.timestamp is None  # caller's object is never mutated (deep copy)
    assert ds.data_definition.timestamp == "execution_timestamp"


def test_strict_without_definition_still_auto_generates():
    _, cur = _make_frames()
    ds = PandasDataset(cur, strict=True)
    assert ds.data_definition.get_column_type("feat") == ColumnType.Numerical
    assert ds.data_definition.get_column_type("target") == ColumnType.Categorical


def test_from_pandas_plumbs_strict():
    _, cur = _make_frames()
    definition = DataDefinition(numerical_columns=["feat"], categorical_columns=["target"])

    ds = Dataset.from_pandas(cur, data_definition=definition, strict=True)

    assert ds.data_definition.timestamp is None
    assert ds.data_definition.get_column_type("execution_timestamp") == ColumnType.Unknown


def test_explicit_strict_false_matches_default():
    _, cur = _make_frames()
    definition = DataDefinition(numerical_columns=["feat"], categorical_columns=["target"])
    ds = PandasDataset(cur, data_definition=definition, strict=False)
    assert ds.data_definition.timestamp == "execution_timestamp"
