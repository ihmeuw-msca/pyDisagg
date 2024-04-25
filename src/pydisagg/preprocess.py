from typing import Tuple

import numpy as np
import pandas as pd

from pydisagg.age_split.age_var import (
    age_id_map,
    match_cols,
    match_pats,
    match_pops,
)


def validate_data(df, pop_df, age_id_map=age_id_map):
    """
    Validates the input dataframe by checking for NaN values, invalid sex_id, invalid location_ids, and invalid year_ids.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to be validated.
    age_id_map : dict
        A dictionary mapping age_ids to their corresponding values.
    pop_df : pandas.DataFrame
        The population dataframe used for validation.

    Returns
    -------
    valid_df : pandas.DataFrame
        The validated dataframe with invalid rows removed.
    nan_df : pandas.DataFrame
        A dataframe containing the rows with validation errors and their corresponding error reasons.
    """

    if "year_id" in df.columns:
        df["input_year_id"] = df["year_id"]

    df["year_id"] = np.ceil((df["year_start"] + df["year_end"]) / 2).astype(int)

    # Create a dataframe for NaN values
    nan_df1 = df[df.isna().any(axis=1)].copy()
    nan_df1["error_reason"] = (
        "NaN values in the following columns: "
        + ", ".join(nan_df1.columns[nan_df1.isna().any()].tolist())
    )

    # Check if 'sex_id' is a string, if so, convert to lowercase and replace 'male' with 1 and 'female' with 2
    if df["sex_id"].dtype == "object":
        df["sex_id"] = (
            df["sex_id"]
            .apply(lambda x: x.lower() if isinstance(x, str) else x)
            .replace({"male": 1, "female": 2})
        )

    nan_df2 = df[~df["sex_id"].isin([1, 2])].copy()
    nan_df2["error_reason"] = "sex_id is not M/F or 1/2"

    # Filter df for invalid location_ids and add error_reason
    nan_df3 = df[~df["location_id"].isin(pop_df["location_id"])].copy()
    nan_df3["error_reason"] = "location_id is not in pop_df"

    # Filter df for invalid year_ids and add error_reason
    nan_df4 = df[~df["year_id"].isin(pop_df["year_id"])].copy()
    nan_df4["error_reason"] = "year_id is not in pop_df"

    # Concatenate all error dataframes
    nan_df = pd.concat([nan_df1, nan_df2, nan_df3, nan_df4], ignore_index=True)

    # Get the 'row_id' values in nan_df
    row_ids_to_drop = nan_df["row_id"]

    # Find the indices of the rows in df that have the same 'row_id' as in nan_df
    indices_to_drop = df[df["row_id"].isin(row_ids_to_drop)].index

    # Drop these indices from df to create valid_df
    valid_df = df.drop(indices_to_drop)

    # valid_df['value'] = valid_df['value']+ 1e-10

    return valid_df, nan_df


def expand_row(row: pd.Series, df_age_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Expands a single row of data based on age group criteria.

    Parameters:
        row (pd.Series): The input row to be expanded.
        df_age_groups (pd.DataFrame): The DataFrame containing age group information.

    Returns:
        pd.DataFrame: The expanded DataFrame with additional rows based on age group criteria.
    """

    # Create 'age_mid' column
    df_age_groups["age_mid"] = (
        df_age_groups["age_group_years_start"]
        + df_age_groups["age_group_years_end"]
    ) / 2
    """
    # Filter df_age_groups
    filtered_df = df_age_groups[
        (df_age_groups["age_group_years_start"] < row["original_data_age_end"])
        & (df_age_groups["age_group_years_end"] > row["original_data_age_start"])
    ]
     Also tried >= and <
    """
    filtered_df = df_age_groups[
        (
            (
                df_age_groups["age_group_years_start"]
                >= row["original_data_age_start"]
            )
            & (
                df_age_groups["age_group_years_start"]
                < row["original_data_age_end"]
            )
        )
        | (
            (
                df_age_groups["age_group_years_end"]
                > row["original_data_age_start"]
            )
            & (
                df_age_groups["age_group_years_end"]
                <= row["original_data_age_end"]
            )
        )
        | (
            (
                df_age_groups["age_group_years_start"]
                <= row["original_data_age_start"]
            )
            & (
                df_age_groups["age_group_years_end"]
                >= row["original_data_age_end"]
            )
        )
    ]

    """
    # If filtered_df is empty, return an empty DataFrame
    if filtered_df.empty:
        print(
            f"Row with index {row.name} resulted in an empty DataFrame after filtering. Row details:\n{row}"
        )
        return pd.DataFrame()
    """

    # Repeat the input row for the number of rows in filtered_df
    repeated_row = pd.DataFrame([row] * len(filtered_df)).reset_index(drop=True)

    # Add the relevant 'age_group_id' to each row
    repeated_row["age_group_id"] = filtered_df["age_group_id"].values
    repeated_row["age_mid"] = filtered_df["age_mid"].values

    repeated_row["age_group_years_start"] = filtered_df[
        "age_group_years_start"
    ].values
    repeated_row["age_group_years_end"] = filtered_df[
        "age_group_years_end"
    ].values

    return repeated_row


def merge_pops(df_input, df_pop, df_nan=None):
    """
    Merge population data with input data based on specified columns.

    Parameters:
    - df_input (pandas.DataFrame): Input data to be merged.
    - df_pop (pandas.DataFrame): Population data to be merged.
    - df_nan (pandas.DataFrame, optional): DataFrame to store unmatched rows.

    Returns:
    - df_merged (pandas.DataFrame): Merged data with population information.
    - df_nan (pandas.DataFrame): DataFrame containing unmatched rows.

    """
    # Define the columns to match on
    match_columns = ["age_group_id", "location_id", "year_id", "sex_id"]

    df_pop_subset = df_pop[match_columns + ["population"]]
    df_merged = pd.merge(df_input, df_pop_subset, on=match_columns, how="left")

    # Find rows where the merge was not successful (i.e., population is NaN)
    unmatched_rows = df_merged[df_merged["population"].isna()].copy()

    # If df_nan is not provided, create it
    if df_nan is None:
        df_nan = pd.DataFrame(
            columns=df_input.columns.tolist() + ["error_reason"]
        )

    # Add the unmatched rows to df_nan
    unmatched_rows["error_reason"] = unmatched_rows[match_columns].apply(
        lambda x: ", ".join([f"{col} is missing" for col in x.index[x.isna()]])
        if x.isna().any()
        else f"populations missing {', '.join([f'{col}: {x[col]}' for col in x.index if x[col] not in df_pop[col].values])}",
        axis=1,
    )

    # Drop the 'population' column from unmatched_rows
    unmatched_rows = unmatched_rows.drop(columns="population")

    # Add the unmatched rows to df_nan
    df_nan = pd.concat([df_nan, unmatched_rows], ignore_index=True)

    # Remove the unmatched rows from df_merged
    df_merged = df_merged.dropna(subset=["population"])

    return df_merged, df_nan


def merge_patterns(df_input, df_nan, patterns, drop_col):
    """
    Merge the 'df_input' and 'patterns' dataframes based on 'sex_id' and 'age_group_id'.
    Handle missing values in the merged dataframe and update the 'df_nan' dataframe.

    Parameters:
    - df_input (pandas.DataFrame): Input dataframe to be merged.
    - df_nan (pandas.DataFrame): Dataframe to store rows with missing values.
    - patterns (pandas.DataFrame): Dataframe containing patterns to be merged.
    - drop_col (bool): Flag indicating whether to drop the 'draw_' columns from 'patterns'.

    Returns:
    - df_input (pandas.DataFrame): Merged dataframe with NaN values dropped.
    - df_nan (pandas.DataFrame): Updated dataframe with rows containing missing values.
    """

    # Create a new column 'mean_draw' in the 'patterns' dataframe
    draw_cols = [col for col in patterns.columns if col.startswith("draw_")]
    patterns["mean_draw"] = patterns[draw_cols].mean(axis=1)

    keep_cols = ["sex_id", "age_group_id", "mean_draw"]
    # If drop_col is True, drop the other draw_ columns
    if drop_col:
        draw_cols = [col for col in patterns.columns if col.startswith("draw_")]
        patterns = patterns.drop(columns=draw_cols)
    else:
        keep_cols += draw_cols

    # Merge 'df_input' and 'patterns' dataframes on 'sex_id' and 'age_group_id'
    patterns_subset = patterns[keep_cols]
    df_merged = pd.merge(
        df_input, patterns_subset, on=["sex_id", "age_group_id"], how="left"
    )

    # Specify the columns to check for NaN values
    columns_to_check = ["sex_id", "population", "age_group_id"]

    # Identify the rows with NaN values in the specified columns
    missing_rows = df_merged[
        df_merged[columns_to_check].isnull().any(axis=1)
    ].copy()

    missing_rows["error_reason"] = missing_rows.apply(
        lambda row: (
            "sex_id is not 1 or 2"
            if pd.isnull(row["sex_id"]) or row["sex_id"] not in [1, 2]
            else "age_group_id"
        ),
        axis=1,
    )

    # Add these rows to 'df_nan' dataframe
    df_nan = pd.concat([df_nan, missing_rows], ignore_index=True)

    # Drop rows with NaN values in the specified columns from df_merged
    df_input = df_merged.dropna(subset=columns_to_check)

    # Check if all age_group_ids in df_input are in patterns
    missing_age_group_ids = set(df_input["age_group_id"]) - set(
        patterns["age_group_id"]
    )
    if missing_age_group_ids:
        missing_rows = df_input[
            df_input["age_group_id"].isin(missing_age_group_ids)
        ].copy()
        missing_rows["error_reason"] = "patterns missing age_group_id"
        df_nan = pd.concat([df_nan, missing_rows], ignore_index=True)
        df_input = df_input[
            ~df_input["age_group_id"].isin(missing_age_group_ids)
        ]

    return df_input, df_nan


def process_data(
    df_input: pd.DataFrame,
    pops: pd.DataFrame,
    patterns: pd.DataFrame,
    df_age_groups: pd.DataFrame = age_id_map,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the input data by expanding rows, merging populations, and merging patterns.

    Args:
        df_input (pd.DataFrame): The input dataframe.
        df_age_groups (pd.DataFrame): The dataframe containing age groups.
        pops (pd.DataFrame): The dataframe containing population data.
        patterns (pd.DataFrame): The dataframe containing patterns data.

    Returns:
        pd.DataFrame: The expanded dataframe.
        pd.DataFrame: The dataframe containing dropped rows.

    """
    missing_columns = {}

    for col in match_cols:
        if col not in df_input.columns:
            if "df_input" not in missing_columns:
                missing_columns["df_input"] = []
            missing_columns["df_input"].append(col)

    for col in match_pops:
        if col not in pops.columns:
            if "pops" not in missing_columns:
                missing_columns["pops"] = []
            missing_columns["pops"].append(col)

    for col in match_pats:
        if col not in patterns.columns:
            if "patterns" not in missing_columns:
                missing_columns["patterns"] = []
            missing_columns["patterns"].append(col)

    if missing_columns:
        error_message = ""
        for df, cols in missing_columns.items():
            error_message += (
                f"Dataframe {df} is missing the column(s): {', '.join(cols)}\n"
            )
        raise KeyError(error_message)

    valid_df, df_nan = validate_data(df_input, pops, df_age_groups)
    df_expanded = pd.concat(
        valid_df.apply(expand_row, df_age_groups=df_age_groups, axis=1).tolist()
    ).reset_index(drop=True)
    df_expanded, df_nan = merge_pops(df_expanded, pops, df_nan)
    df_expanded, df_nan = merge_patterns(
        df_input=df_expanded, patterns=patterns, df_nan=df_nan, drop_col=True
    )
    # Maybe include unique locations?

    # Print the number of rows dropped
    print(
        f"Number of rows dropped: {df_nan.shape[0]} out of {len(df_expanded)} total"
    )

    # Print the number of unique 'row_id' dropped
    print(f"Number of 'row_id' dropped: {df_nan['row_id'].nunique()}")

    missing_value_counts = df_nan["error_reason"].value_counts()
    unique_nid_per_missing_value = df_nan.groupby("error_reason")[
        "row_id"
    ].nunique()

    for missing_value, count in missing_value_counts.items():
        unique_nid = unique_nid_per_missing_value[missing_value]
        # Print the number of rows dropped for each unique type of 'error_reason'
        print(
            f"{count} rows dropped because of {missing_value}, corresponding to {unique_nid} row_id's"
        )

    return df_expanded, df_nan
