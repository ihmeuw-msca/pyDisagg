import pandas as pd
import numpy as np
import itertools
from pydisagg.ihme.splitter import (
    SexDataConfig,
    SexPatternConfig,
    SexPopulationConfig,
    SexSplitter,
    AgeDataConfig,
    AgePatternConfig,
    AgePopulationConfig,
    AgeSplitter,
)

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
    list(itertools.product([1, 2], range(21), [2020, 2021])),
    columns=["sex_id", "age_group_id", "year_id"],
)
age_pattern["age_group_years_start"] = 19 + age_pattern["age_group_id"]
age_pattern["age_group_years_end"] = age_pattern["age_group_years_start"] + 1
for i in range(3):
    age_pattern[f"draw_{i}"] = np.tile(
        [0.2 + i * 0.1, 0.3 + i * 0.1, 0.4 + i * 0.1], int(len(age_pattern) / 3)
    )

sex_pop = pd.DataFrame(
    {
        "location_id": [34, 34, 34, 34, 6, 6, 6, 6] * 2,
        "year_id": [2020, 2020, 2021, 2021, 2020, 2020, 2021, 2021] * 2,
        "sex_id": [1, 2] * 8,
        "population": [5000, 5200, 5300, 5500, 6000, 6200, 6300, 6500] * 2,
    }
)

age_pop = pd.DataFrame(
    list(itertools.product([34, 6], [2020, 2021], [1, 2], range(21))),
    columns=["location_id", "year_id", "sex_id", "age_group_id"],
)
age_pop["population"] = np.tile([1000, 1050, 1100, 1150, 1200], int(len(age_pop) / 5))

# Merging and cleaning up population data
pop = pd.merge(sex_pop, age_pop, on=["location_id", "year_id", "sex_id"])
pop = pop.drop(columns=["population_y"]).rename(columns={"population_x": "population"})

# Sex splitting (example configuration, adjust as needed)
sex_splitter = SexSplitter(
    data=SexDataConfig(
<<<<<<< HEAD
        index=[
            "nid",
            "seq",
            "location_id",
            "year_id",
            "sex_id",
            "age_lwr",
            "age_upr",
        ],
=======
        index=["nid", "seq", "location_id", "year_id", "sex_id", "age_lwr", "age_upr"],
>>>>>>> 886e366 (Updated documentation)
        val="val",
        val_sd="val_sd",
    ),
    pattern=SexPatternConfig(by=["year_id"], val="draw_mean", val_sd="draw_sd"),
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

# Age splitting (example configuration, adjust as needed)
age_splitter = AgeSplitter(
    data=AgeDataConfig(
        index=["nid", "seq", "location_id", "year_id", "sex_id"],
        age_lwr="age_lwr",
        age_upr="age_upr",
        val="val",  # Assuming this should be the result from sex splitting
        val_sd="val_sd",
    ),
    pattern=AgePatternConfig(
        by=["sex_id", "year_id"],
        age_key="age_group_id",
        age_lwr="age_group_years_start",
        age_upr="age_group_years_end",
        draws=[f"draw_{i}" for i in range(3)],  # Adjust based on actual draw columns
    ),
    population=AgePopulationConfig(
<<<<<<< HEAD
        index=["age_group_id", "location_id", "year_id", "sex_id"],
        val="population",
=======
        index=["age_group_id", "location_id", "year_id", "sex_id"], val="population"
>>>>>>> 886e366 (Updated documentation)
    ),
)

result_age_sex_df = age_splitter.split(
    data=result_sex_df, pattern=age_pattern, population=age_pop
)
