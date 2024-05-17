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
        raise ValueError(f"{name} has negative values in: {negative}")


def validate_interval(
    df: DataFrame, lwr: str, upr: str, index: list[str], name: str
) -> None:
    invalid_index = pd.MultiIndex.from_frame(
        df.query(f"{lwr} >= {upr}")[index]
    ).to_list()
    if invalid_index:
        raise ValueError(
            f"{name} has invalid interval with index: {invalid_index}"
        )


def validate_noindexdiff(
    df_ref: DataFrame, df: DataFrame, index: list[str], name: str
) -> None:
    index_ref = pd.MultiIndex.from_frame(df_ref[index])
    index = pd.MultiIndex.from_frame(df[index])
    missing_index = index_ref.difference(index).to_list()

    if missing_index:
        raise ValueError(f"Missing {name} info for index: {missing_index}")
