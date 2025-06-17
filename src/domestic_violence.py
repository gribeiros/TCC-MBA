"""Utility functions for the domestic violence dataset."""

from pathlib import Path
import pandas as pd


def load_yearly_files(data_dir: str | Path) -> pd.DataFrame:
    """Load all annual CSV files into a single DataFrame.

    Parameters
    ----------
    data_dir : str or Path
        Path to the folder containing ``violencia_domestica_YYYY.csv`` files.

    Returns
    -------
    DataFrame
        Concatenation of all years with an extra ``year`` column.
    """
    data_dir = Path(data_dir)
    frames = []
    for path in sorted(data_dir.glob("violencia_domestica_*.csv")):
        year = int(path.stem.split("_")[-1])
        df = pd.read_csv(path, sep=";")
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def format_date(df: pd.DataFrame, column: str = "data_fato") -> pd.DataFrame:
    """Convert the date column to ``datetime``.

    Invalid or malformed values become ``NaT``.
    """
    df[column] = pd.to_datetime(df[column], errors="coerce", yearfirst=True)
    return df
