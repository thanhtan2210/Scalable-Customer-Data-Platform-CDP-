import logging
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def set_log_level():
    """Set a default log level for tests so test output is stable."""
    logging.basicConfig(level=logging.ERROR)


@pytest.fixture
def tmp_df():
    """Return a small DataFrame used in tests."""
    return pd.DataFrame({"Total Charges": ["100", "200", "300"], "Monthly Charges": [50.0, 75.0, 100.0], "Tenure Months": [1, 10, 24], "Partner": ["Yes", "No", "Yes"]})


@pytest.fixture
def tmp_csv(tmp_path, tmp_df):
    """Write the tmp_df to a CSV file in tmp_path and return the path."""
    p = tmp_path / "sample.csv"
    tmp_df.to_csv(p, index=False)
    return p
