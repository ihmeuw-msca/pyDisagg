from pydisagg.age_split.age_var import rename_dict_dis


def rename_df(frozen_df, rename_dict=rename_dict_dis, drop=True):
    """
    Renames columns of a DataFrame based on a given dictionary and optionally drops columns.

    Parameters:
        frozen_df (DataFrame): The input DataFrame to be renamed.
        rename_dict (dict): A dictionary mapping old column names to new column names.
        drop (bool, optional): Whether to drop columns not present in the rename_dict.
                               Defaults to True.

    Returns:
        return_df: The renamed dataframe
        frozen_df: The original dataframe with 'row_id' column added
    """
    # Create a copy of the input DataFrame to avoid changing it in place
    df = frozen_df.copy()
    frozen_df_copy = frozen_df.copy()

    df.insert(0, "row_id", range(1, len(df) + 1))
    frozen_df_copy.insert(0, "row_id", range(1, len(frozen_df_copy) + 1))

    return_df = df.rename(columns=rename_dict)
    if drop:
        return_df = return_df[list(rename_dict.values())]
    return_df.insert(0, "row_id", range(1, len(return_df) + 1))

    return return_df, frozen_df_copy


def glue_back(df, frozen_df):
    """
    Appends the columns of a frozen_df to a df based on the "row_id" column.

    Parameters:
        df (DataFrame): The main DataFrame to which columns will be appended.
        frozen_df (DataFrame): The DataFrame whose columns will be appended to df.

    Returns:
        DataFrame: The merged dataframe
    """
    merged_df = df.merge(
        frozen_df, on="row_id", how="left", suffixes=("", "_frozen")
    )
    merged_df = merged_df.filter(regex="^(?!.*_frozen)")

    return merged_df
