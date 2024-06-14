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
        raise ValueError(f"{name} has NaN values in columns: {nan_columns}")


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
