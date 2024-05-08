import pandas as pd
import pytest
from pydisagg.ihme.age_var import rename_dict_dis
from pydisagg.ihme.helper import rename_df, glue_back


def test_rename_df():
    # Prepare a test dataframe
    df = pd.DataFrame(
        {
            "year_start": [2000, 2001],
            "year_end": [2002, 2003],
            "location_id": [1, 2],
            "sex": ["male", "female"],
            "age_start": [10, 20],
            "age_end": [30, 40],
            "mean": [15, 25],
            "standard_error": [0.1, 0.2],
        }
    )

    # Call the function
    renamed_df, original_df = rename_df(df, rename_dict=rename_dict_dis)

    # Check if the columns are correctly renamed
    assert list(renamed_df.columns) == [
        "row_id",
        "year_start",
        "year_end",
        "location_id",
        "sex_id",
        "original_data_age_start",
        "original_data_age_end",
        "value",
        "SE",
    ]

    # Check if 'row_id' column is correctly added
    assert "row_id" in renamed_df.columns
    assert "row_id" in original_df.columns

    # Check if 'sex_id' column is correctly handled
    assert renamed_df["sex_id"].dtype != "object"
    assert all(renamed_df["sex_id"].isin([1, 2, 3]))


def test_rename_df_missing_key():
    # Prepare a test dataframe
    df = pd.DataFrame(
        {
            "year_start": [2000, 2001],
            "year_end": [2002, 2003],
            "location_id": [1, 2],
            "sex": ["male", "female"],
            "age_start": [10, 20],
            "age_end": [30, 40],
            "mean": [15, 25],
        }
    )  # 'standard_error' column is missing

    # Check if a ValueError is raised when a key in the rename_dict is not present in the dataframe
    with pytest.raises(ValueError):
        rename_df(df, rename_dict=rename_dict_dis)


def test_glue_back():
    # Prepare test dataframes
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "year_start": [2000, 2001, 2002],
            "year_end": [2002, 2003, 2004],
            "location_id": [1, 2, 3],
        }
    )

    frozen_df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "sex_id": ["male", "female", "both"],
            "original_data_age_start": [10, 20, 30],
            "original_data_age_end": [30, 40, 50],
            "value": [15, 25, 35],
            "SE": [0.1, 0.2, 0.3],
        }
    )

    # Call the function
    merged_df = glue_back(df, frozen_df)

    # Check if the dataframes are correctly merged
    assert list(merged_df.columns) == [
        "row_id",
        "year_start",
        "year_end",
        "location_id",
        "sex_id",
        "original_data_age_start",
        "original_data_age_end",
        "value",
        "SE",
    ]

    # Check if the columns with "_frozen" suffix are correctly filtered out
    assert not any("_frozen" in col for col in merged_df.columns)
