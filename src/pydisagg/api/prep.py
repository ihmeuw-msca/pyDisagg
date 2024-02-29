import pandas as pd
import numpy as np
from pydisagg.models import RateMultiplicativeModel
from pydisagg.models import LogOdds_model
from pydisagg.disaggregate import split_datapoint
import pandas as pd


def check_dataframe(df, new_column_list=None):
    column_list = [
        "nid",
        "year_start",
        "year_end",
        "location_id",
        "sex",
        "o_age_start",
        "o_age_end",
        "value",
        "SE",
    ]

    if new_column_list is not None:
        if len(new_column_list) != len(column_list):
            print(
                "Error: The provided list does not match the length of the default column list."
            )
            return df

        column_mapping = dict(zip(new_column_list, column_list))
        df.rename(columns=column_mapping, inplace=True)

    missing_columns = [col for col in column_list if col not in df.columns]

    if missing_columns:
        print(f"The following columns are missing: {missing_columns}")

    df_error = pd.DataFrame()

    if "sex_id" not in df.columns:
        sex_mapping = {"Male": 1, "Female": 2}
        df["sex_id"] = df["sex"].map(sex_mapping)

        df_error = df[df["sex_id"].isna()].copy()
        df_error["missing_reason"] = "Sex not M/F"
        dropped_rows = df_error.shape[0]
        # if dropped_rows > 0:
        # print(f"Dropped {dropped_rows} rows because 'Sex not M/F'")
        df = df.dropna(subset=["sex_id"])

    df = df.copy()
    df.loc[:, "value"] = df["value"] + 1e-16
    df_copy = df.copy()
    df_copy["year_id"] = np.ceil(
        (df_copy["year_start"] + df_copy["year_end"]) / 2
    ).astype(int)
    df_copy = df_copy.dropna(subset=["sex_id"])

    for col in column_list:
        missing_rows = df_copy[df_copy[col].isna()].copy()
        missing_rows["missing_reason"] = f"missing {col} column"
        df_error = pd.concat([df_error, missing_rows])
        df_copy = df_copy.dropna(subset=[col])
        if len(missing_rows) > 0:
            print(f"Dropped {len(missing_rows)} rows because of missing data in {col}")

    if "age_group_id" in df_copy.columns:
        df_copy = df_copy.rename(columns={"age_group_id": "orig_age_group_id"})

    # print(f"After check_dataframe, the shape of df is {df_copy.shape}")
    # print(f"After check_dataframe, the shape of df_error is {df_error.shape}")

    return df_copy, df_error


def expand_row(row: pd.Series, df_age_groups: pd.DataFrame) -> pd.DataFrame:
    # Create 'age_mid' column
    df_age_groups["age_mid"] = (
        df_age_groups["age_group_years_start"] + df_age_groups["age_group_years_end"]
    ) / 2

    # Filter df_age_groups
    filtered_df = df_age_groups[
        (df_age_groups["age_group_years_start"] <= row["o_age_end"])
        & (df_age_groups["age_group_years_end"] >= row["o_age_start"])
    ]

    # If filtered_df is empty, return an empty DataFrame
    if filtered_df.empty:
        print(
            f"Row with index {row.name} resulted in an empty DataFrame after filtering. Row details:\n{row}"
        )
        return pd.DataFrame()

    # Repeat the input row for the number of rows in filtered_df
    repeated_row = pd.DataFrame([row] * len(filtered_df)).reset_index(drop=True)

    # Add the relevant 'age_group_id' to each row
    repeated_row["age_group_id"] = filtered_df["age_group_id"].values
    repeated_row["age_mid"] = filtered_df["age_mid"].values

    repeated_row["age_group_years_start"] = filtered_df["age_group_years_start"].values
    repeated_row["age_group_years_end"] = filtered_df["age_group_years_end"].values

    return repeated_row


def merge_pops(df_input, df_pop, df_nan=None):
    # Define the columns to match on
    match_columns = ["age_group_id", "location_id", "year_id", "sex_id"]

    # Merge df_input and df_pop on the match_columns
    df_merged = pd.merge(
        df_input, df_pop[match_columns + ["population"]], on=match_columns, how="left"
    )
    # print(f"After merging, the shape of df_merged is {df_merged.shape}")

    # Find rows where the merge was not successful (i.e., population is NaN)
    unmatched_rows = df_merged[df_merged["population"].isna()].copy()
    # print(f"Number of unmatched rows: {unmatched_rows.shape[0]}")

    # If df_nan is not provided, create it
    if df_nan is None:
        df_nan = pd.DataFrame(columns=df_input.columns.tolist() + ["missing_reason"])

    # Add the unmatched rows to df_nan
    unmatched_rows["missing_reason"] = unmatched_rows.apply(
        lambda x: ", ".join(x.index[x.isna()]), axis=1
    )
    df_nan = pd.concat([df_nan, unmatched_rows], ignore_index=True)

    # Remove the unmatched rows from df_merged
    df_merged = df_merged.dropna(subset=["population"])
    # print(f"After dropping NaN population, the shape of df_merged is {df_merged.shape}")

    # print(f"After merge_pops, the shape of df_merged is {df_merged.shape}")
    # print(f"After merge_pops, the shape of df_nan is {df_nan.shape}")
    return df_merged, df_nan


def merge_patterns(df_input, df_nan, patterns, drop_col):
    # print(f"Before merge_patterns, the shape of df_input is {df_input.shape}")
    # print(f"Before merge_patterns, the shape of df_nan is {df_nan.shape}")

    # Create a new column 'mean_draw' in the 'patterns' dataframe
    draw_cols = [col for col in patterns.columns if col.startswith("draw_")]
    patterns["mean_draw"] = patterns[draw_cols].mean(axis=1)
    # If drop_col is True, drop the other draw_ columns
    if drop_col:
        draw_cols = [col for col in patterns.columns if col.startswith("draw_")]
        patterns = patterns.drop(columns=draw_cols)

    # Convert 'sex_id' and 'age_group_id' to int in both dataframes
    # df_input[['sex_id', 'age_group_id']] = df_input[['sex_id', 'age_group_id']].astype(int)
    # patterns[['sex_id', 'age_group_id']] = patterns[['sex_id', 'age_group_id']].astype(int)

    # print("Sorted unique sex_id in df_input:", np.sort(df_input['sex_id'].unique()))
    # print("Sorted unique age_group_id in df_input:", np.sort(df_input['age_group_id'].unique()))

    # print("Sorted unique sex_id in patterns:", np.sort(patterns['sex_id'].unique()))
    # print("Sorted unique age_group_id in patterns:", np.sort(patterns['age_group_id'].unique()))

    # Merge 'df_input' and 'patterns' dataframes on 'sex_id' and 'age_group_id'
    df_merged = pd.merge(df_input, patterns, on=["sex_id", "age_group_id"], how="left")

    # print(f"Directly After merge_patterns, the shape of df_input is {df_merged.shape}")

    # Specify the columns to check for NaN values
    columns_to_check = ["sex_id", "population"]

    # Identify the rows with NaN values in the specified columns
    missing_rows = df_merged[df_merged[columns_to_check].isnull().any(axis=1)].copy()

    missing_rows["missing_reason"] = missing_rows.apply(
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

    # print(f"After merge_patterns, the shape of df_input is {df_input.shape}")
    # print(f"After merge_patterns, the shape of df_nan is {df_nan.shape}")

    return df_input, df_nan


def process_data(
    df_input: pd.DataFrame,
    df_age_groups: pd.DataFrame,
    pops: pd.DataFrame,
    patterns: pd.DataFrame,
    new_column_list=None,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_input, df_nan = check_dataframe(df_input, new_column_list)
    df_expanded = pd.concat(
        df_input.apply(expand_row, df_age_groups=df_age_groups, axis=1).tolist()
    ).reset_index(drop=True)
    # print(f"After expanision, the shape of df_expanded is {df_expanded.shape}")

    df_expanded, df_nan = merge_pops(df_expanded, pops, df_nan)

    df_expanded, df_nan = merge_patterns(
        df_input=df_expanded, patterns=patterns, df_nan=df_nan, drop_col=True
    )

    # Print the number of rows dropped
    print(f"Number of rows dropped: {df_nan.shape[0]}")

    # Print the number of unique 'nid' dropped
    print(f"Number of 'nid' dropped: {df_nan['nid'].nunique()}")
    # print("\n")
    # Print the number of rows dropped for each unique type of 'missing_reason'
    missing_value_counts = df_nan["missing_reason"].value_counts()
    unique_nid_per_missing_value = df_nan.groupby("missing_reason")["nid"].nunique()
    for missing_value, count in missing_value_counts.items():
        unique_nid = unique_nid_per_missing_value[missing_value]
        # print("\n")
        # print(f"{count} rows dropped because of {missing_value}, corresponding to {unique_nid} nid's")

    # print(f"After process_data, the shape of df_input is {df_input.shape}")
    # print(f"After process_data, the shape of df_expanded is {df_expanded.shape}")
    # print(f"After process_data, the shape of df_nan is {df_nan.shape}")

    return df_input, df_expanded, df_nan


def split_row(row, df_expanded, match_cols):
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

    ages_included = np.arange(row["o_age_start"], row["o_age_end"])
    patterns_interp = np.interp(
        ages_included, output_subset["age_mid"], output_subset["mean_draw"]
    )
    pops_interp = np.interp(
        ages_included, output_subset["age_mid"], output_subset["population"]
    )

    split_result, SE = split_datapoint(
        observed_total=row["value"],
        bucket_populations=pops_interp,
        rate_pattern=patterns_interp,
        model=RateMultiplicativeModel(),
        output_type="rate",
        normalize_pop_for_average_type_obs=True,
        observed_total_se=row["SE"],
    )

    split_result_df = pd.DataFrame(
        {
            "age_val": ages_included,
            "split_result": split_result,
            "pop": pops_interp,
            "split_result_SE": SE,
        }
    )
    split_results = [
        split_result_df[split_result_df["age_val"].between(l, u, inclusive="left")]
        for l, u in zip(
            output_subset["age_group_years_start"], output_subset["age_group_years_end"]
        )
    ]

    # Update post_split_prev and post_split_SE with checks for population sum
    output_subset["post_split_prev"] = [
        (
            (x["split_result"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else np.nan
        )
        for x in split_results
    ]
    output_subset["post_split_SE"] = [
        (
            (x["split_result_SE"] * x["pop"]).sum() / x["pop"].sum()
            if x["pop"].sum() > 0
            else np.nan
        )
        for x in split_results
    ]

    return output_subset


def split_df(df_expanded: pd.DataFrame, df_nan: pd.DataFrame = None):
    df_split_result = df_expanded.copy()
    df_split_result["split_result"] = np.nan

    match_cols = [
        "nid",
        "year_start",
        "year_end",
        "o_age_start",
        "o_age_end",
        "sex_id",
        "value",
        "SE",
    ]
    obs_to_split = df_expanded[match_cols].drop_duplicates()

    row_split_func = lambda x: split_row(x, df_expanded, match_cols=match_cols)

    try:
        result = pd.concat(
            obs_to_split.apply(row_split_func, axis=1).tolist(), ignore_index=True
        )
    except Exception as e:
        print("Error occurred:", e)
        result = None

    # print(f"After split_df, the shape of result is {result.shape if result is not None else 'None'}")
    return result


###########################################################################################

df_input_raw = pd.read_csv("fresh_data/dismod/dismod_shape_bv_tosplit.csv", index_col=0)
df_age_groups = pd.read_csv("py_temp/age_group_data.csv")
df_age_groups = df_age_groups[~df_age_groups["age_group_id"].isin([2, 3, 4])]

pops = pd.read_csv(
    "fresh_data/populations.csv",
)
patterns_dis = pd.read_csv("fresh_data/dismod/dismod_agepattern.csv", index_col=0)

df_input_dis, df_expanded_dis, df_nan_out_dis = process_data(
    df_input_raw,
    df_age_groups,
    pops,
    patterns_dis,
    new_column_list=[
        "nid",
        "year_start",
        "year_end",
        "location_id",
        "sex",
        "age_start",
        "age_end",
        "mean",
        "standard_error",
    ],
)

result_dis = split_df(df_expanded_dis, df_nan_out_dis)
result_dis.to_csv("post_dis_prep.csv")
df_nan_out_dis.to_csv("nan_dis_prep.csv")


# column_list = ['nid', 'year_start', 'year_end', 'location_id', 'sex', 'o_age_start','o_age_end', 'value', 'SE']

df_st_raw = pd.read_csv("input_data/stgpr/fpg_pre_split.csv", index_col=0)
df_st = df_st_raw.copy()
df_st["SE"] = df_st["variance"].apply(np.sqrt)
st_patterns = pd.read_csv("input_data/stgpr/fpg_pat.csv", index_col=0)

df_input, df_expanded, df_nan_out = process_data(
    df_st,
    df_age_groups,
    pops,
    st_patterns,
    new_column_list=[
        "nid",
        "orig_year_start",
        "orig_year_end",
        "location_id",
        "sex",
        "orig_age_start",
        "orig_age_end",
        "val",
        "SE",
    ],
)

result = split_df(df_expanded, df_nan_out)
result.to_csv("post_split_data_prep_stgpr.csv")
df_nan_out.to_csv("nan_results_prep_stgpr.csv")
