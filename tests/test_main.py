from pathlib import Path
import pandas as pd

from src.main import run_pipeline


def test_run_pipeline_dry_run(tmp_csv, tmp_path):
    """The pipeline in dry-run mode should return a DataFrame and not write files."""
    out_dir = tmp_path / "out"
    df = run_pipeline(str(tmp_csv), str(out_dir),
                      partition_col=None, dry_run=True)
    # ensure it returns a DataFrame and did not write any files
    assert isinstance(df, pd.DataFrame)
    assert not any(out_dir.iterdir()) if out_dir.exists() else True


def test_run_pipeline_writes_parquet(tmp_csv, tmp_path):
    """The pipeline should write a parquet file when not in dry-run mode."""
    out_dir = tmp_path / "out"
    df = run_pipeline(str(tmp_csv), str(out_dir),
                      partition_col=None, dry_run=False)
    # check that the single parquet file exists and can be read
    single = out_dir / "cleaned_telco.parquet"
    assert single.exists()
    read_df = pd.read_parquet(single)
    assert list(read_df.columns) == list(df.columns)
    # If the pipeline created interval-like fields, they should be converted to str in parquet
    # (e.g., tenure_bin is often intervals.) Ensure parquet read doesn't fail and values are strings
    if "tenure_bin" in read_df.columns:
        # Can be either object (string) or categorical; ensure it's not Interval
        assert not isinstance(read_df["tenure_bin"].dtype, pd.IntervalDtype)
        assert (read_df["tenure_bin"].dtype == object) or isinstance(
            read_df["tenure_bin"].dtype, pd.CategoricalDtype)


def test_save_parquet_partition_conversion_fails(tmp_df, tmp_path):
    """If partition column cannot be converted to datetime, save_parquet should raise."""
    from src.etl.cleaning import save_parquet
    df = tmp_df.copy()
    df["bad_date"] = ["notadate", "stillbad", "yup"]
    out_dir = tmp_path / "out"
    import pytest

    with pytest.raises(Exception):
        save_parquet(df, out_dir, partition_col="bad_date")
