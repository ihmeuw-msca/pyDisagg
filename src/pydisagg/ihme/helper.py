from pydisagg.ihme.age_var import rename_dict_dis
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def rename_df(frozen_df, rename_dict=rename_dict_dis, drop=True):
    """
    Parameters
    ----------
    frozen_df : DataFrame
        The input DataFrame to be renamed.
    rename_dict : dict
        A dictionary mapping old column names to new column names.
    drop : bool, optional
        Whether to drop columns not present in the rename_dict. Defaults to True.

    Returns
    -------
    return_df : DataFrame
        The renamed dataframe
    frozen_df : DataFrame
        The original dataframe with 'row_id' column added
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

    Parameters
    ----------
    df : DataFrame
        The main DataFrame to which columns will be appended.
    frozen_df : DataFrame
        The DataFrame whose columns will be appended to df.

    Returns
    -------
    DataFrame
        The merged dataframe
    """
    merged_df = df.merge(
        frozen_df, on="row_id", how="left", suffixes=("", "_frozen")
    )
    merged_df = merged_df.filter(regex="^(?!.*_frozen)")

    return merged_df


def plot_results(
    result_df, pattern_df, row_id, title="Default Title", y_label="Some Measure"
):
    # Filter result_df for the given row_id
    sub_df = result_df.query(f"row_id == {row_id}")

    # For each row in sub_df
    for _, row in sub_df.iterrows():
        # Plot a line from age_group_years_start to age_group_years_end with the value of post_split_prev
        plt.plot(
            [row["age_group_years_start"], row["age_group_years_end"]],
            [row["post_split_prev"], row["post_split_prev"]],
            color="blue",
            label="post",
        )

        # Add a shaded region around the line to represent the uncertainty from standard error value = post_split_SE
        plt.fill_between(
            [row["age_group_years_start"], row["age_group_years_end"]],
            [
                row["post_split_prev"] - row["post_split_SE"],
                row["post_split_prev"] - row["post_split_SE"],
            ],
            [
                row["post_split_prev"] + row["post_split_SE"],
                row["post_split_prev"] + row["post_split_SE"],
            ],
            color="blue",
            alpha=0.1,
        )

        # Look up the mean_draw value in patterns_df corresponding to the rows age_group_id and sex_id
        mean_draw_df = pattern_df.query(
            " and ".join(
                [f"{col} == {row[col]}" for col in ["age_group_id", "sex_id"]]
            )
        )

        if not mean_draw_df.empty:
            mean_draw = mean_draw_df["mean_draw"].values[0]

            # Plot the mean_draw value and add a legend to label it "pattern"
            plt.plot(
                [row["age_group_years_start"], row["age_group_years_end"]],
                [mean_draw, mean_draw],
                color="red",
                label="pattern",
            )
        else:
            print(
                f"No matching rows in pattern_df for age_group_id {row['age_group_id']} and sex_id {row['sex_id']}"
            )

    single = sub_df.iloc[0]
    # Plot a line for "pre" using the values from the "value" column and the standard error from the "SE" column
    plt.plot(
        [single["original_data_age_start"], row["original_data_age_end"]],
        [row["value"], row["value"]],
        color="green",
        label="pre",
    )

    # Add a shaded region around the "pre" line to represent the uncertainty from standard error value = SE
    plt.fill_between(
        [single["original_data_age_start"], single["original_data_age_end"]],
        [single["value"] - single["SE"], single["value"] - single["SE"]],
        [single["value"] + single["SE"], single["value"] + single["SE"]],
        color="green",
        alpha=0.1,
    )

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    lower = sub_df["age_group_years_start"].min()
    upper = sub_df["age_group_years_end"].max()
    plt.xlim(lower - 2, upper + 2)
    plt.xticks(np.arange(lower, upper + 1, 5))

    # Calculate the range of y values
    y_min = min(
        sub_df["post_split_prev"].min() - sub_df["post_split_SE"].max(),
        sub_df["value"].min() - sub_df["SE"].max(),
        pattern_df["mean_draw"].min(),
    )
    y_max = max(
        sub_df["post_split_prev"].max() + sub_df["post_split_SE"].max(),
        sub_df["value"].max() + sub_df["SE"].max(),
        pattern_df["mean_draw"].max(),
    )

    # Set y-axis range to include all data points with 20% padding
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Set x-axis label to "Age (years)"
    plt.xlabel("Age (years)")

    # Set y-axis label to "Prevalence"
    plt.ylabel(y_label)

    # Set title to "Prevalence by Age Group"
    plt.title(title)

    fig, ax = plt.subplots()

    return fig
