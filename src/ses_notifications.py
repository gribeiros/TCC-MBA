"""Functions for analysing SES notification data."""

import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency


def to_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert notification and birth dates to ``datetime`` objects."""
    df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], dayfirst=True)
    df["DT_NASC"] = pd.to_datetime(df["DT_NASC"], dayfirst=True)
    return df


def chi2_analysis(vars: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise chi-square tests for the categorical columns.

    Parameters
    ----------
    vars : list of str
        Column names to compare.
    df : DataFrame
        Data source containing the variables.

    Returns
    -------
    DataFrame
        Test results sorted by chi-square statistic.
    """
    results = []
    for var1, var2 in combinations(vars, 2):
        table = pd.crosstab(df[var1], df[var2])
        chi2, p, _, _ = chi2_contingency(table)
        results.append({"var1": var1, "var2": var2, "chi2": chi2, "p_value": p})
    return pd.DataFrame(results).sort_values("chi2", ascending=False)
