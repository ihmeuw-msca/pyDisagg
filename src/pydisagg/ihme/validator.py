import numpy as np
import pandas as pd
from pandas import DataFrame


def validate_columns(df: DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{name} has missing columns: {missing}")


def validate_index(df: DataFrame, index: list[str], name: str) -> None:
    duplicated_index = pd.MultiIndex.from_frame(
        df[df[index].duplicated()][index]
    ).to_list()
    if duplicated_index:
        raise ValueError(f"{name} has duplicated index: {duplicated_index}")


def validate_nonan(df: DataFrame, name: str) -> None:
    nan_columns = df.columns[df.isna().any(axis=0)].to_list()
    if nan_columns:
        raise ValueError(f"{name} has NaN values in columns: {nan_columns}")


def validate_positive(
    df: DataFrame, columns: list[str], name: str, strict: bool = True
) -> None:
    op = "<=" if strict else "<"
    negative = [col for col in columns if df.eval(f"{col} {op} 0").any()]
    if negative:
        message = "0 or negative values in" if strict else "negative values in"
        raise ValueError(f"{name} has {message}: {negative}")


def validate_interval(
    df: DataFrame, lwr: str, upr: str, index: list[str], name: str
) -> None:
    invalid_index = pd.MultiIndex.from_frame(
        df.query(f"{lwr} >= {upr}")[index]
    ).to_list()
    if invalid_index:
        raise ValueError(
            f"{name} has invalid interval with index: {invalid_index}. "
            "Lower age must be strictly less than upper age."
        )


def validate_noindexdiff(
    df_ref: DataFrame, df: DataFrame, index: list[str], name: str
) -> None:
    index_ref = pd.MultiIndex.from_frame(df_ref[index])
    index = pd.MultiIndex.from_frame(df[index])
    missing_index = index_ref.difference(index).to_list()

    if missing_index:
        raise ValueError(f"Missing {name} info for index: {missing_index}")


def validate_pat_coverage(
    df: DataFrame,
    lwr: str,
    upr: str,
    pat_lwr: str,
    pat_upr: str,
    index: list[str],
    name: str,
) -> None:
    """Validation checks for incomplete age pattern
    * pattern age intervals do not overlap or have gaps
    * smallest pattern interval doesn't cover the left end point of data
    * largest pattern interval doesn't cover the right end point of data
    """
    # sort dataframe
    df = df.sort_values(index + [lwr, upr, pat_lwr, pat_upr], ignore_index=True)
    df_group = df.groupby(index)

    # check overlap or gap in pattern
    shifted_pat_upr = df_group[pat_upr].shift(1)
    connect_index = shifted_pat_upr.notnull()
    connected = np.allclose(
        shifted_pat_upr[connect_index], df.loc[connect_index, pat_lwr]
    )

    if not connected:
        raise ValueError(
            f"{name} pattern has overlap or gap between the lower and upper "
            "bounds across categories."
        )

    # check coverage of head and tail
    head_covered = df_group.first().eval(f"{lwr} >= {pat_lwr}").all()
    tail_covered = df_group.last().eval(f"{upr} <= {pat_upr}").all()

    if not (head_covered and tail_covered):
        raise ValueError(
            f"{name} pattern does not cover the data lower and/or upper bound"
        )
