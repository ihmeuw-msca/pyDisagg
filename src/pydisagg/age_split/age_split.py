
"""
TODO: 
    Move this into pyDisagg
    Refactor to not change things in place
    Wrap up preprocessing into one function (that can call the others)
    Maybe put helper functions for preprocessing into another module
        Make them have data in format on left of rename_dis (the input to rename df)
        Don't make them invent a dictionary for renaming, but make them match a standard input for now
    Put global variables into another .py file that we import
    TODO Later: Add asympotic uncertainty from pattern (option for multiple of identity in log space maybe)
    Instantiate and call model objects instead of strings, allow for defaults
        Optional: Allow strings, but mainly work with actual model object
"""


import pandas as pd
import numpy as np
from pydisagg.models import RateMultiplicativeModel
from pydisagg.models import LogOdds_model
from pydisagg.disaggregate import split_datapoint
import pandas as pd
import matplotlib.pyplot as plt
#TODO Handle low age groups more elegantly, we're just doing 1 year buckets for now, we should have an exception for age groups under 1, and explicitly model those as single

from pydisagg.age_split.age_var import age_id_map, rename_dict_dis, match_cols

def rename_df(frozen_df, rename_dict = rename_dict_dis, drop=True):
    """
    Renames columns of a DataFrame based on a given dictionary and optionally drops columns.
    
    Parameters:
        frozen_df (DataFrame): The input DataFrame to be renamed.
        rename_dict (dict): A dictionary mapping old column names to new column names.
        drop (bool, optional): Whether to drop columns not present in the rename_dict. 
                               Defaults to True.
    
    Returns:
        input_df: The renamed dataframe
        frozen_df: The original dataframe
    """
    frozen_df.insert(0, 'row_id', range(1, len(frozen_df) + 1))
    input_df = frozen_df.rename(columns=rename_dict)
    if drop:
        input_df = input_df[list(rename_dict.values())]
    input_df.insert(0, 'row_id', range(1, len(input_df) + 1)) 
    
    # Add the 'year_id' column
    input_df['year_id'] = np.ceil((input_df['year_start'] + input_df['year_end']) / 2).astype(int)
    
    return input_df, frozen_df

def validate_data(df, age_id_map = age_id_map, pop_df):
    """
    Validates the input dataframe by checking for NaN values, invalid sex_id, invalid location_ids, and invalid year_ids.
    
    Args:
        df (pandas.DataFrame): The input dataframe to be validated.
        age_id_map (dict): A dictionary mapping age_ids to their corresponding values.
        pop_df (pandas.DataFrame): The population dataframe used for validation.
        
    Returns:
        valid_df (pandas.DataFrame): The validated dataframe with invalid rows removed.
        nan_df (pandas.DataFrame): A dataframe containing the rows with validation errors and their corresponding error reasons.
    """
    # Create a dataframe for NaN values
    nan_df1 = df[df.isna().any(axis=1)].copy()
    nan_df1['error_reason'] = "NaN values in the following columns: " + ', '.join(nan_df1.columns[nan_df1.isna().any()].tolist())

    # Create a dataframe for invalid sex_id
    if df['sex_id'].dtype == 'object':
        df['sex_id'] = df['sex_id'].str.lower()
        
    nan_df2 = df[~df['sex_id'].isin(['male', 'female', 1, 2])].copy()

    # Assign error reason for invalid 'sex_id'
    nan_df2['error_reason'] = "sex_id is not M/F or 1/2"

    # Create a copy of df for sex_id validation
    df_sex_id = df.copy()

    # Check if 'sex_id' is a string, if so, convert to lowercase and replace 'male' with 1 and 'female' with 2
    df_sex_id['sex_id'] = df_sex_id['sex_id'].apply(lambda x: x.lower() if isinstance(x, str) else x).replace({'male': 1, 'female': 2})

    # Replace df['sex_id'] with the validated df_sex_id['sex_id']
    df['sex_id'] = df_sex_id['sex_id']

    # Create dataframes for invalid location_ids and year_ids
    unique_location_ids = df['location_id'].unique()
    unique_year_ids = df['year_id'].unique()
    
    # Filter df for invalid location_ids and add error_reason
    nan_df3 = df[~df['location_id'].isin(pop_df['location_id'])].copy()
    nan_df3['error_reason'] = 'location_id is not in pop_df'

    # Filter df for invalid year_ids and add error_reason
    nan_df4 = df[~df['year_id'].isin(pop_df['year_id'])].copy()
    nan_df4['error_reason'] = 'year_id is not in pop_df'

    # Concatenate all error dataframes
    nan_df = pd.concat([nan_df1, nan_df2, nan_df3, nan_df4], ignore_index=True)

    # Get the 'row_id' values in nan_df
    row_ids_to_drop = nan_df['row_id']

    # Find the indices of the rows in df that have the same 'row_id' as in nan_df
    indices_to_drop = df[df['row_id'].isin(row_ids_to_drop)].index

    # Drop these indices from df to create valid_df
    valid_df = df.drop(indices_to_drop)

    #valid_df['value'] = valid_df['value']+ 1e-10
    
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
        df_age_groups["age_group_years_start"] + df_age_groups["age_group_years_end"]
    ) / 2
    '''
    # Filter df_age_groups
    filtered_df = df_age_groups[
        (df_age_groups["age_group_years_start"] < row["original_data_age_end"])
        & (df_age_groups["age_group_years_end"] > row["original_data_age_start"])
    ]
     Also tried >= and <
    '''
    filtered_df = df_age_groups[
        ((df_age_groups["age_group_years_start"] >= row["original_data_age_start"]) & (df_age_groups["age_group_years_start"] < row["original_data_age_end"]))
        | ((df_age_groups["age_group_years_end"] > row["original_data_age_start"]) & (df_age_groups["age_group_years_end"] <= row["original_data_age_end"]))
        | ((df_age_groups["age_group_years_start"] <= row["original_data_age_start"]) & (df_age_groups["age_group_years_end"] >= row["original_data_age_end"]))
    ]
    
    
    '''
    # If filtered_df is empty, return an empty DataFrame
    if filtered_df.empty:
        print(
            f"Row with index {row.name} resulted in an empty DataFrame after filtering. Row details:\n{row}"
        )
        return pd.DataFrame()
    '''
    
    # Repeat the input row for the number of rows in filtered_df
    repeated_row = pd.DataFrame([row] * len(filtered_df)).reset_index(drop=True)

    # Add the relevant 'age_group_id' to each row
    repeated_row["age_group_id"] = filtered_df["age_group_id"].values
    repeated_row["age_mid"] = filtered_df["age_mid"].values

    repeated_row["age_group_years_start"] = filtered_df["age_group_years_start"].values
    repeated_row["age_group_years_end"] = filtered_df["age_group_years_end"].values

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
    df_merged = pd.merge(
        df_input, df_pop_subset, on=match_columns, how="left"
    )

    # Find rows where the merge was not successful (i.e., population is NaN)
    unmatched_rows = df_merged[df_merged["population"].isna()].copy()

    # If df_nan is not provided, create it
    if df_nan is None:
        df_nan = pd.DataFrame(columns=df_input.columns.tolist() + ["error_reason"])

    # Add the unmatched rows to df_nan
    unmatched_rows["error_reason"] = unmatched_rows[match_columns].apply(
    lambda x: ", ".join([f"{col} is missing" for col in x.index[x.isna()]]) 
    if x.isna().any() 
    else f"populations missing {', '.join([f'{col}: {x[col]}' for col in x.index if x[col] not in df_pop[col].values])}", 
    axis=1
    )
    
    # Drop the 'population' column from unmatched_rows
    unmatched_rows = unmatched_rows.drop(columns='population')

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
    df_merged = pd.merge(df_input, patterns_subset, on=["sex_id", "age_group_id"], how="left")

    # Specify the columns to check for NaN values
    columns_to_check = ["sex_id", "population","age_group_id"]

    # Identify the rows with NaN values in the specified columns
    missing_rows = df_merged[df_merged[columns_to_check].isnull().any(axis=1)].copy()

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
    missing_age_group_ids = set(df_input["age_group_id"]) - set(patterns["age_group_id"])
    if missing_age_group_ids:
        missing_rows = df_input[df_input["age_group_id"].isin(missing_age_group_ids)].copy()
        missing_rows["error_reason"] = "patterns missing age_group_id"
        df_nan = pd.concat([df_nan, missing_rows], ignore_index=True)
        df_input = df_input[~df_input["age_group_id"].isin(missing_age_group_ids)]

    return df_input, df_nan

def process_data(
    df_input: pd.DataFrame,
    df_age_groups: pd.DataFrame,
    pops: pd.DataFrame,
    patterns: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Process the input data by expanding rows, merging populations, and merging patterns.

    Args:
        df_input (pd.DataFrame): The input dataframe.
        df_age_groups (pd.DataFrame): The dataframe containing age groups.
        pops (pd.DataFrame): The dataframe containing population data.
        patterns (pd.DataFrame): The dataframe containing patterns data.
        new_column_list (list, optional): List of new column names. Defaults to None.

    Returns:
        pd.DataFrame: The processed input dataframe.
        pd.DataFrame: The expanded dataframe.
        pd.DataFrame: The dataframe containing dropped rows.

    """
    valid_df, df_nan = validate_data(df_input, df_age_groups, pops)
    df_expanded = pd.concat(
        valid_df.apply(expand_row, df_age_groups=df_age_groups, axis=1).tolist()
    ).reset_index(drop=True)

    df_expanded, df_nan = merge_pops(df_expanded, pops, df_nan)

    df_expanded, df_nan = merge_patterns(
        df_input=df_expanded, patterns=patterns, df_nan=df_nan, drop_col=False
    )

    # Print the number of rows dropped
    print(f"Number of rows dropped: {df_nan.shape[0]} out of {len(df_expanded)} total")

    # Print the number of unique 'row_id' dropped
    print(f"Number of 'row_id' dropped: {df_nan['row_id'].nunique()}")

    missing_value_counts = df_nan["error_reason"].value_counts()
    unique_nid_per_missing_value = df_nan.groupby("error_reason")["row_id"].nunique()
    for missing_value, count in missing_value_counts.items():
        unique_nid = unique_nid_per_missing_value[missing_value]
        # Print the number of rows dropped for each unique type of 'error_reason'
        print(f"{count} rows dropped because of {missing_value}, corresponding to {unique_nid} row_id's")

    return df_input, df_expanded, df_nan

def split_row(row, df_expanded, match_cols = match_cols, model="Rate",pattern_col = 'mean_draw'):
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

    #ages_included = np.arange(row["original_data_age_start"], row["original_data_age_end"]+ 1)
    #TODO Verify this, This should be right exclusive
    ages_included = np.arange(row["original_data_age_start"], row["original_data_age_end"])


    patterns_interp = np.interp(
        ages_included, output_subset["age_mid"], output_subset[pattern_col]
    )

    #TODO Write sum consistent interpolator, will look like deconvolution
    #TODO Smoothest thing that satisfies discrete sum constraints
    pops_interp = np.interp(
        ages_included, output_subset["age_mid"], output_subset["population"]/(output_subset['age_group_years_end'] - output_subset['age_group_years_start'])
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
    '''
    split_results = [
        split_result_df[split_result_df["age_val"].between(l, u, inclusive="left")]
        for l, u in zip(
            output_subset["age_group_years_start"], output_subset["age_group_years_end"]
        )
    ]
    '''

    #We're relying on indexing order here, and it can be brittle
    #We should explicitly use keys or something in the future
    split_results = [
        split_result_df[split_result_df["age_val"].between(max(l, row["original_data_age_start"]), min(u, row["original_data_age_end"]), inclusive="left")]
        for l, u in zip(
            output_subset["age_group_years_start"], output_subset["age_group_years_end"]
        )
    ]

    # Update post_split_prev and post_split_SE with checks for population sum
    output_subset["post_split_prev"] = [
        (
            (x["split_result"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else 1000
        )
        for x in split_results
    ]

    output_subset["adj_pop"] = [
        (
            x["pop"].sum()
        )
        for x in split_results
    ]
    output_subset["post_split_SE"] = [
        (
            (x["split_result_SE"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else 1000
        )
        for x in split_results
    ]
    
    # Create a new column 'age_group_low' with values from 'age_group_years_start' 
    # but replace values that are less than original_data_age_start with original_data_age_start
    age_group_low = output_subset['age_group_years_start'].copy()
    age_group_low.loc[age_group_low < row['original_data_age_start']] = row['original_data_age_start']

    # Get the index of the 'age_group_years_end' column
    idx = output_subset.columns.get_loc('age_group_years_end')

    # Insert the new column after 'age_group_years_end'
    output_subset.insert(idx + 1, 'age_bin_low', age_group_low)
       
    # Create a new column 'age_group_high' with values from 'age_group_years_end' 
    # but replace values that are greater than original_data_age_end with original_data_age_end
    age_group_high = output_subset['age_group_years_end'].copy()
    age_group_high.loc[age_group_high > row['original_data_age_end']] = row['original_data_age_end']

    # Insert the new column after 'age_group_low'
    output_subset.insert(idx + 2, 'age_bin_high', age_group_high)

    
    # Drop the age_mid column
    #output_subset = output_subset.drop(columns=['age_mid'])
    
    return output_subset

def split_df(df_expanded: pd.DataFrame, df_nan: pd.DataFrame = None, model: str = "rate"):
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

    row_split_func = lambda x: split_row(x, df_expanded, match_cols=match_cols, model=model.lower())

    try:
        result = pd.concat(
            obs_to_split.apply(row_split_func, axis=1).tolist(), ignore_index=True
        )
    except Exception as e:
        print("Error occurred:", e)
        result = None
        
    return result