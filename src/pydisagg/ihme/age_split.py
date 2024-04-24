"""
TODO:
    ## Move this into pyDisagg ##
    ## Refactor to not change things in place ##
    Wrap up preprocessing into one function (that can call the others)
    Maybe put helper functions for preprocessing into another module
        Make them have data in format on left of rename_dis (the input to rename df)
        Don't make them invent a dictionary for renaming, but make them match a standard input for now
    Use np.nan instead of filling in a number (-1000)
    ## Put global variables into another .py file that we import ##
    TODO Later: Add asympotic uncertainty from pattern (option for multiple of identity in log space maybe)
    Instantiate and call model objects instead of strings, allow for defaults
        Optional: Allow strings, but mainly work with actual model object
    TODO Handle low age groups more elegantly, we're just doing 1 year buckets for now, we should have an exception for age groups under 1, and explicitly model those as single
"""

import numpy as np
import pandas as pd

from .age_var import match_cols
from ..disaggregate import split_datapoint
from ..models import LogOdds_model, RateMultiplicativeModel
from typing import Optional


def split_row(
    row,
    df_expanded,
    match_cols=match_cols,
    model="Rate",
    pattern_col="mean_draw",
):
    """
    Splits a row of data based on age groups and performs interpolation.

    Args:
        row (pd.Series): The row of data to be split.
        df_expanded (pd.DataFrame): The expanded dataframe containing age group information.
        match_cols (list): The list of columns used to match rows in df_expanded.
        model (str): The model to be used for splitting. Can be "LogOdds" or "Rate".

    Returns:
        pd.DataFrame: The resulting split data.

    Raises:
        ValueError: If the required columns 'age_group_years_start' or 'age_group_years_end' are missing in df_expanded.
    """
    model = model.lower()

    if (
        "age_group_years_start" not in df_expanded.columns
        or "age_group_years_end" not in df_expanded.columns
    ):
        print(
            "Error: Required columns 'age_group_years_start' or 'age_group_years_end' are missing in df_expanded"
        )
        return pd.DataFrame()

    output_subset = df_expanded[
        (df_expanded[match_cols] == row).all(axis=1)
    ].sort_values("age_mid")

    # ages_included = np.arange(row["original_data_age_start"], row["original_data_age_end"]+ 1)
    # TODO Verify this, This should be right exclusive
    ages_included = np.arange(
        row["original_data_age_start"], row["original_data_age_end"]
    )

    patterns_interp = np.interp(
        ages_included, output_subset["age_mid"], output_subset[pattern_col]
    )

    # TODO Write sum consistent interpolator, will look like deconvolution
    # TODO Smoothest thing that satisfies discrete sum constraints
    pops_interp = np.interp(
        ages_included,
        output_subset["age_mid"],
        output_subset["population"]
        / (
            output_subset["age_group_years_end"]
            - output_subset["age_group_years_start"]
        ),
    )
    if row["value"] == 0:
        split_result, SE = 0, 0
    else:
        if model == "logodds":
            try:
                split_result, SE = split_datapoint(
                    observed_total=row["value"],
                    bucket_populations=pops_interp,
                    rate_pattern=patterns_interp,
                    model=LogOdds_model(),
                    output_type="rate",
                    normalize_pop_for_average_type_obs=True,
                    observed_total_se=row["SE"],
                )
            except Exception as e:
                print(f"Error in split_datapoint with LogOdds model: {e}")
                split_result, SE = np.nan, np.nan
        elif model == "rate":
            try:
                split_result, SE = split_datapoint(
                    observed_total=row["value"],
                    bucket_populations=pops_interp,
                    rate_pattern=patterns_interp,
                    model=RateMultiplicativeModel(),
                    output_type="rate",
                    normalize_pop_for_average_type_obs=True,
                    observed_total_se=row["SE"],
                )
            except Exception as e:
                print(f"Error in split_datapoint with Rate model: {e}")
                split_result, SE = np.nan, np.nan
        else:
            print("Error: Invalid model specified")
            return pd.DataFrame()

    split_result_df = pd.DataFrame(
        {
            "age_val": ages_included,
            "split_result": split_result,
            "pop": pops_interp,
            "split_result_SE": SE,
        }
    )
    """
    split_results = [
        split_result_df[split_result_df["age_val"].between(l, u, inclusive="left")]
        for l, u in zip(
            output_subset["age_group_years_start"], output_subset["age_group_years_end"]
        )
    ]
    """

    # We're relying on indexing order here, and it can be brittle
    # We should explicitly use keys or something in the future
    split_results = [
        split_result_df[
            split_result_df["age_val"].between(
                max(low, row["original_data_age_start"]),
                min(up, row["original_data_age_end"]),
                inclusive="left",
            )
        ]
        for low, up in zip(
            output_subset["age_group_years_start"],
            output_subset["age_group_years_end"],
        )
    ]

    # Update post_split_prev and post_split_SE with checks for population sum
    output_subset["post_split_prev"] = [
        (
            (x["split_result"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else -1000
        )
        for x in split_results
    ]

    output_subset["adj_pop"] = [(x["pop"].sum()) for x in split_results]
    output_subset["post_split_SE"] = [
        (
            (x["split_result_SE"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else -1000
        )
        for x in split_results
    ]

    # Create a new column 'age_group_low' with values from 'age_group_years_start'
    # but replace values that are less than original_data_age_start with original_data_age_start
    age_group_low = output_subset["age_group_years_start"].copy()
    age_group_low.loc[age_group_low < row["original_data_age_start"]] = row[
        "original_data_age_start"
    ]

    # Get the index of the 'age_group_years_end' column
    idx = output_subset.columns.get_loc("age_group_years_end")
    output_subset.insert(idx + 1, "age_bin_low", age_group_low)

    # Create a new column 'age_group_high' with values from 'age_group_years_end'
    # but replace values that are greater than original_data_age_end with original_data_age_end
    age_group_high = output_subset["age_group_years_end"].copy()
    age_group_high.loc[age_group_high > row["original_data_age_end"]] = row[
        "original_data_age_end"
    ]
    output_subset.insert(idx + 2, "age_bin_high", age_group_high)

    return output_subset


def split_df(
    df_expanded: pd.DataFrame, df_nan: pd.DataFrame = None, model: str = "rate"
):
    """
    Splits a DataFrame based on specified columns and returns the result.

    Args:
        df_expanded (pd.DataFrame): The DataFrame to be split.
        df_nan (pd.DataFrame, optional): Another DataFrame. Defaults to None.
        model (str, optional): The model to be used for splitting. Can be "rate" or "logodds". Defaults to "rate".

    Returns:
        pd.DataFrame: The resulting split DataFrame.
    """
    df_split_result = df_expanded.copy()
    df_split_result["split_result"] = np.nan

    match_cols = [
        "row_id",
        "year_start",
        "year_end",
        "original_data_age_start",
        "original_data_age_end",
        "sex_id",
        "value",
        "SE",
    ]
    obs_to_split = df_expanded[match_cols].drop_duplicates()

    def row_split_func(x):
        return split_row(
            x, df_expanded, match_cols=match_cols, model=model.lower()
        )

    try:
        result = pd.concat(
            obs_to_split.apply(row_split_func, axis=1).tolist(),
            ignore_index=True,
        )
    except Exception as e:
        print("Error occurred:", e)
        result = None

    return result
