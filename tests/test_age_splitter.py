import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError
from pydisagg.ihme.splitter import (
    AgeSplitter,
    AgeDataConfig,
    AgePatternConfig,
    AgePopulationConfig,
)

from pydisagg.ihme.validator import (
    validate_positive,
    validate_noindexdiff,
)


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def splitter(data, pattern, population):
    pattern_config = AgePatternConfig(
        by=["sex_id", "location_id", "year_id"],
        age_key="age_group_id",
        age_lwr="age_start",
        age_upr="age_end",
        draws=["draw_0", "draw_1", "draw_2"],
    )
    data_config = AgeDataConfig(
        index=["uid", "sex_id", "location_id", "year_id"],
        age_lwr="age_start",
        age_upr="age_end",
        val="val",
        val_sd="val_sd",
    )
    population_config = AgePopulationConfig(
        index=["sex_id", "location_id", "age_group_id", "year_id"], val="population"
    )
    return AgeSplitter(
        data=data_config, pattern=pattern_config, population=population_config
    )


def test_parse_data_success(splitter, data):
    # Should pass without exceptions
    parsed_data = splitter.parse_data(data, positive_strict=True)
    assert isinstance(parsed_data, pd.DataFrame)
    assert not parsed_data.empty


def test_parse_data_missing_column(splitter, data):
    # Remove a required column
    data = data.drop(columns=["val"])
    with pytest.raises(KeyError, match="has missing columns"):
        splitter.parse_data(data, positive_strict=True)


def test_parse_data_missing_index_column(splitter, data):
    # Remove a required index column
    data = data.drop(columns=["uid"])
    with pytest.raises(KeyError, match="has missing columns"):
        splitter.parse_data(data, positive_strict=True)


def test_parse_data_nan_values(splitter, data):
    # Introduce NaN values
    data.loc[0, "val"] = np.nan
    with pytest.raises(ValueError, match="has NaN values"):
        splitter.parse_data(data, positive_strict=True)


def test_parse_data_negative_val_sd(splitter, data):
    # Introduce negative values in val_sd
    data.loc[0, "val_sd"] = -1.0
    with pytest.raises(ValueError, match="has 0 or negative values"):
        splitter.parse_data(data, positive_strict=True)


def test_parse_data_invalid_intervals(splitter, data):
    # Introduce invalid interval (age_start >= age_end)
    data.loc[0, "age_start"] = 20
    data.loc[0, "age_end"] = 10
    with pytest.raises(ValueError, match="has invalid interval"):
        splitter.parse_data(data, positive_strict=True)


def test_parse_pattern_success(splitter, data, pattern):
    # Ensure the pattern parsing works correctly with valid input
    parsed_pattern = splitter.parse_pattern(data, pattern, positive_strict=True)
    assert isinstance(parsed_pattern, pd.DataFrame)
    assert not parsed_pattern.empty


def test_parse_pattern_missing_columns(splitter, data, pattern):
    # Ensure 'val' and 'val_sd' are set to the expected columns in the splitter config
    splitter.pattern.val = "val"
    splitter.pattern.val_sd = "val_sd"

    # Ensure 'val' and 'val_sd' are missing
    pattern = pattern.drop(columns=["val", "val_sd"], errors="ignore")

    # Check that 'draws' are present
    assert all(
        col in pattern.columns for col in splitter.pattern.draws
    ), "Draw columns are missing"

    # This should generate 'val' and 'val_sd' from the 'draws' or raise a ValueError
    if not splitter.pattern.draws:
        with pytest.raises(ValueError, match="Must provide draws for pattern"):
            splitter.parse_pattern(data, pattern, positive_strict=True)
    else:
        parsed_pattern = splitter.parse_pattern(data, pattern, positive_strict=True)
        assert "val" in parsed_pattern.columns
        assert "val_sd" in parsed_pattern.columns


def test_parse_pattern_missing_draws(splitter, data, pattern):
    # Remove both val and val_sd, and ensure pattern.draws is empty
    splitter.pattern.val = "nonexistent_column"
    splitter.pattern.val_sd = "nonexistent_column"
    splitter.pattern.draws = []
    with pytest.raises(ValueError, match="Must provide draws for pattern"):
        splitter.parse_pattern(data, pattern, positive_strict=True)


def test_parse_pattern_missing_index(splitter, data, pattern):
    # Ensure the 'location_id' column exists before dropping it
    assert "location_id" in pattern.columns
    pattern = pattern.drop(columns=["location_id"])
    with pytest.raises(KeyError, match="has missing columns"):
        splitter.parse_pattern(data, pattern, positive_strict=True)


def test_parse_pattern_invalid_interval(splitter, data, pattern):
    # Introduce invalid intervals in the pattern
    pattern.loc[0, "age_start"] = 20
    pattern.loc[0, "age_end"] = 10
    with pytest.raises(ValueError, match="has invalid interval"):
        splitter.parse_pattern(data, pattern, positive_strict=True)


def test_parse_pattern_pat_coverage(splitter, data, pattern):
    # Modify pattern intervals so they don't cover the data intervals
    pattern.loc[0, "age_start"] = 50
    pattern.loc[0, "age_end"] = 60
    with pytest.raises(ValueError, match="pattern does not cover the data"):
        splitter.parse_pattern(data, pattern, positive_strict=True)


def test_parse_pattern_nan_values(splitter, data, pattern):
    # Ensure that the 'val' column is generated from draws if it doesn't exist
    if "val" not in pattern.columns or "val_sd" not in pattern.columns:
        pattern["val"] = pattern[splitter.pattern.draws].mean(axis=1)
        pattern["val_sd"] = pattern[splitter.pattern.draws].std(axis=1)

    # Now that 'val' exists, ensure the column can accept NaN values
    pattern["val"] = pattern["val"].astype(float)

    # Introduce NaN values in the 'val' column
    pattern.loc[0, "val"] = np.nan

    # Manually check if NaN is correctly set
    assert pd.isna(pattern.loc[0, "val"]), "NaN not correctly set in 'val' column"

    # Ensure validate_nonan is called in parse_pattern
    with pytest.raises(ValueError, match="has NaN values"):
        splitter.parse_pattern(data, pattern, positive_strict=True)


def test_validate_positive():
    df = pd.DataFrame({"val_sd": [-1.0, -0.5, 0.0, 1.0, 2.0]})

    try:
        validate_positive(df, ["val_sd"], "Test Pattern", strict=False)
    except ValueError as e:
        assert "negative values in" in str(e)
    else:
        pytest.fail("Expected ValueError not raised by validate_positive")


def test_validate_noindexdiff():
    # Create two DataFrames with different indices
    df_ref = pd.DataFrame(
        {
            "uid": [0, 1, 2],
            "sex_id": [1, 1, 2],
            "location_id": [1, 1, 2],
            "year_id": [2010, 2010, 2011],
            "age_start": [0, 5, 10],
        }
    )

    df = pd.DataFrame(
        {
            "uid": [0, 1],
            "sex_id": [1, 1],
            "location_id": [1, 1],
            "year_id": [2010, 2010],
            "age_start": [0, 5],
        }
    )

    try:
        validate_noindexdiff(
            df_ref, df, ["uid", "sex_id", "location_id", "year_id"], "Test Data"
        )
    except ValueError as e:
        assert "Missing Test Data info for" in str(e)
    else:
        pytest.fail("Expected ValueError not raised by validate_noindexdiff")
