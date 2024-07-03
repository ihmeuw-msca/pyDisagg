import pandas as pd
import numpy as np
from itertools import product
from pydisagg.ihme.splitter import (
    SexSplitter,
    SexDataConfig,
    SexPatternConfig,
    SexPopulationConfig,
)

# Assuming pyDisagg is correctly installed and the necessary imports are done

# Creating example dataframes
pre_split = pd.DataFrame(
    {
        "nid": [10101, 22222],
        "seq": [1, 2],
        "location_id": [34, 6],
        "year_id": [2020, 2021],
        "sex_id": [3, 3],
        "age_lwr": [20, 25],
        "age_upr": [30, 35],
        "val": [0.5, 0.6],
        "val_sd": [0.1, 0.2],
    }
)

sex_pattern = pd.DataFrame(
    {
        "year_id": range(2016, 2023),
        "draw_mean": [0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0],
        "draw_sd": [0.1, 0.15, 0.12, 0.14, 0.11, 0.13, 0.12],
    }
)

age_pattern = pd.DataFrame(
    list(product([1, 2], range(21), [2020, 2021])),
    columns=["sex_id", "age_group_id", "year_id"],
)
age_pattern["age_group_years_start"] = 19 + age_pattern["age_group_id"]
age_pattern["age_group_years_end"] = age_pattern["age_group_years_start"] + 1
age_pattern["draw_0"] = np.tile([0.2, 0.3, 0.4], len(age_pattern) // 3)
age_pattern["draw_1"] = np.tile([0.5, 0.6, 0.7], len(age_pattern) // 3)
age_pattern["draw_2"] = np.tile([0.8, 0.9, 0.95], len(age_pattern) // 3)

sex_pop = pd.DataFrame(
    {
        "location_id": [34, 34, 34, 34, 6, 6, 6, 6],
        "year_id": [2020, 2020, 2021, 2021, 2020, 2020, 2021, 2021],
        "sex_id": [1, 2, 1, 2, 1, 2, 1, 2],
        "population": [5000, 5200, 5300, 5500, 6000, 6200, 6300, 6500],
    }
)

age_pop = pd.DataFrame(
    list(product([34, 6], [2020, 2021], [1, 2], range(21))),
    columns=["location_id", "year_id", "sex_id", "age_group_id"],
)
# Adjusting the population assignment to match the length of age_pop's index
age_pop["population"] = np.tile([1000, 1050, 1100, 1150, 1200], len(age_pop) // 5 + 1)[
    : len(age_pop)
]

pop = pd.merge(sex_pop, age_pop, on=["location_id", "year_id", "sex_id"])
pop = pop.drop(columns=["population_y"]).rename(columns={"population_x": "population"})

# Sex splitting
sex_splitter = SexSplitter(
    data=SexDataConfig(
        index=["nid", "seq", "location_id", "year_id", "sex_id", "age_lwr", "age_upr"],
        val="val",
        val_sd="val_sd",
    ),
    pattern=SexPatternConfig(
        by=["year_id"],
        val="draw_mean",
        val_sd="draw_sd",
    ),
    population=SexPopulationConfig(
        index=["location_id", "year_id"],
        sex="sex_id",
        sex_m=1,
        sex_f=2,
        val="population",
    ),
)

# Assuming sex_pre_split is defined similarly to pre_split
result_sex_df = sex_splitter.split(
    data=pre_split, pattern=sex_pattern, population=sex_pop
)

print(result_sex_df.head())
