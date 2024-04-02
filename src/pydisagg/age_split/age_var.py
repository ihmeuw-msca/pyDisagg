import numpy as np
import pandas as pd

age_group_id = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    30,
    31,
    32,
    235,
]
age_group_years_start = [
    0,
    0.01917808,
    0.07671233,
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
]
age_group_years_end = [
    0.01917808,
    0.07671233,
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    125,
]
# TODO Handle low age groups more elegantly, we're just doing 1 year buckets for now, we should have an exception for age groups under 1, and explicitly model those as single

age_mid = (np.array(age_group_years_start) + np.array(age_group_years_end)) / 2

age_id_map = pd.DataFrame(
    {
        "age_group_id": age_group_id,
        "age_group_years_start": age_group_years_start,
        "age_group_years_end": age_group_years_end,
        "age_mid": age_mid,
    }
)

rename_dict_dis = {
    "year_start": "year_start",
    "year_end": "year_end",
    "location_id": "location_id",
    "sex": "sex_id",
    "age_start": "original_data_age_start",
    "age_end": "original_data_age_end",
    "mean": "value",
    "standard_error": "SE",
}

rename_dict_stgpr = {
    "orig_year_start": "year_start",
    "orig_year_end": "year_end",
    "location_id": "location_id",
    "sex": "sex_id",
    "orig_age_start": "original_data_age_start",
    "orig_age_end": "original_data_age_end",
    "val": "value",
    "SE": "SE",
}

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

match_pops = ["age_group_id", "location_id", "year_id", "sex_id", "population"]

match_pats = ["sex_id", "age_group_id"]  # , "location_id", "year_id" in the future
