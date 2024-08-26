import numpy as np
import pandas as pd
from pandas import DataFrame


def validate_columns(df: DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        error_message = (
            f"{name} has missing columns: {len(missing)} columns are missing.\n"
        )
        error_message += f"Missing columns: {', '.join(missing)}\n"
        if len(missing) > 5:
            error_message += "First 5 missing columns: \n"
        error_message += ", \n".join(missing[:5])
        error_message += "\n"
        raise KeyError(error_message)


def validate_index(df: DataFrame, index: list[str], name: str) -> None:
    duplicated_index = pd.MultiIndex.from_frame(
        df[df[index].duplicated()][index]
    ).to_list()
    if duplicated_index:
        error_message = f"{name} has duplicated index with {len(duplicated_index)} indices \n"
        error_message += f"Index columns: ({', '.join(index)})\n"
        if len(duplicated_index) > 5:
            error_message += "First 5: \n"
        error_message += ", \n".join(str(idx) for idx in duplicated_index[:5])
        error_message += "\n"
        raise ValueError(error_message)


def validate_nonan(df: DataFrame, name: str) -> None:
    nan_columns = df.columns[df.isna().any(axis=0)].to_list()
    if nan_columns:
        error_message = (
            f"{name} has NaN values in {len(nan_columns)} columns. \n"
        )
        error_message += f"Columns with NaN values: {', '.join(nan_columns)}\n"
        if len(nan_columns) > 5:
            error_message += "First 5 columns with NaN values: \n"
        error_message += ", \n".join(nan_columns[:5])
        error_message += "\n"
        raise ValueError(error_message)


def validate_positive(
    df: DataFrame, columns: list[str], name: str, strict: bool = False
) -> None:
    """Validates that observation values in cols are non-negative or strictly positive"""
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
        error_message = f"{name} has invalid interval with {len(invalid_index)} indices. \nLower age must be strictly less than upper age.\n"
        error_message += f"Index columns: ({', '.join(index)})\n"
        if len(invalid_index) > 5:
            error_message += "First 5 indices with invalid interval: \n"
        error_message += ", \n".join(str(idx) for idx in invalid_index[:5])
        error_message += "\n"
        raise ValueError(error_message)


def validate_noindexdiff(
    df_ref: DataFrame, df: DataFrame, index: list[str], name: str
) -> None:
    index_ref = pd.MultiIndex.from_frame(df_ref[index])
    index = pd.MultiIndex.from_frame(df[index])
    missing_index = index_ref.difference(index).to_list()

    if missing_index:
        error_message = (
            f"Missing {name} info for {len(missing_index)} indices \n"
        )
        error_message += f"Index columns: ({', '.join(index.names)})\n"
        if len(missing_index) > 5:
            error_message += "First 5: \n"
        error_message += ", \n".join(str(idx) for idx in missing_index[:5])
        error_message += "\n"
        raise ValueError(error_message)


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


def validate_realnumber(df: DataFrame, columns: list[str], name: str) -> None:
    """
    Validates that observation values in columns are real numbers and non-zero.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data to validate.
    columns : list of str
        A list of column names to validate within the DataFrame.
    name : str
        A string representing the name of the data or dataset
        (used for constructing error messages).

    Raises
    ------
    ValueError
        If any column contains values that are not real numbers or are zero.
    """
    # Check for non-real or zero values in the specified columns
    invalid = [
        col
        for col in columns
        if not df[col]
        .apply(lambda x: isinstance(x, (int, float)) and x != 0)
        .all()
    ]

    if invalid:
        raise ValueError(f"{name} has non-real or zero values in: {invalid}")
