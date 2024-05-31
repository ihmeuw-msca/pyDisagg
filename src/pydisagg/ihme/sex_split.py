import numpy as np
import pandas as pd

from pydisagg.disaggregate import split_datapoint
from pydisagg.models import RateMultiplicativeModel


def data():
    np.random.seed(123)
    return pd.DataFrame(
        dict(
            uid=range(10),
            sex_id=[1] * 5 + [2] * 5,
            location_id=[1, 2] * 5,
            year_id=[2010] * 10,
            age_start=[0, 5, 10, 17, 20] * 2,
            age_end=[12, 10, 22, 21, 25] * 2,
            val=5.0,
            val_sd=1.0,
        )
    )


def pattern():
    np.random.seed(123)
    pattern_df1 = pd.DataFrame(
        dict(
            sex_id=[1] * 5 + [2] * 5,
            age_start=[0, 5, 10, 15, 20] * 2,
            age_end=[5, 10, 15, 20, 25] * 2,
            age_group_id=list(range(5)) * 2,
            draw_0=np.random.rand(10),
            draw_1=np.random.rand(10),
            draw_2=np.random.rand(10),
            year_id=[2010] * 10,
            location_id=[1] * 10,
        )
    )
    pattern_df2 = pattern_df1.copy()
    pattern_df2["location_id"] = 2
    return pd.concat([pattern_df1, pattern_df2]).reset_index(drop=True)


def population():
    np.random.seed(123)
    sex_id = pd.DataFrame(dict(sex_id=[1, 2]))
    year_id = pd.DataFrame(dict(year_id=[2010]))
    location_id = pd.DataFrame(dict(location_id=[1, 2]))
    age_group_id = pd.DataFrame(dict(age_group_id=range(5)))

    population = (
        sex_id.merge(location_id, how="cross")
        .merge(age_group_id, how="cross")
        .merge(year_id, how="cross")
    )
    population["population"] = 1000
    return population


df = data()
df_pattern = pattern()
df_pop = population()
df = df.drop("sex_id", axis=1)
df["male_pop"] = np.random.uniform(0.5, 2, 10)
df["female_pop"] = np.random.uniform(0.5, 2, 10)
df["prev_ratio_2_over_1"] = np.linspace(1, 2, 10)
df["prev_ratio_se"] = np.random.uniform(0.1, 0.2, 10)


def sex_split_df(df, ratio_2_over_1_col, ratio_se_col, val_col, val_se_col):
    def sex_split_row(row):
        split_result, SE = split_datapoint(
            observed_total=row[val_col],
            bucket_populations=np.array([row["male_pop"], row["female_pop"]]),
            rate_pattern=np.array([1.0, row[ratio_2_over_1_col]]),
            model=RateMultiplicativeModel(),
            output_type="rate",
            normalize_pop_for_average_type_obs=True,
            observed_total_se=row[val_se_col],
            pattern_covariance=np.diag(np.array([0, row[ratio_se_col] ** 2])),
        )
        return pd.Series(
            {
                "split_val_male": split_result[0],
                "split_val_female": split_result[1],
                "se_male": SE[0],
                "se_female": SE[1],
            }
        )

    # Apply the function across the DataFrame
    split_results = df.apply(sex_split_row, axis=1)

    # Create new DataFrames for male and female results
    split_df_male = df.copy()
    split_df_female = df.copy()

    split_df_male[[val_col, val_se_col]] = split_results[
        ["split_val_male", "se_male"]
    ]
    split_df_female[[val_col, val_se_col]] = split_results[
        ["split_val_female", "se_female"]
    ]

    split_df_male["sex_id"] = 1
    split_df_female["sex_id"] = 2

    # Combine the results back into one DataFrame
    final_split_df = (
        pd.concat([split_df_male, split_df_female], ignore_index=True)
        .sort_values(["uid", "sex_id"])
        .reset_index(drop=True)
    )
    return final_split_df


df = sex_split_df(df, "prev_ratio_2_over_1", "prev_ratio_se", "val", "val_sd")
print(df)
