import pytest
import pandas as pd
from pydantic import ValidationError
from pydisagg.ihme.splitter import (
    SexSplitter,
    SexDataConfig,
    SexPatternConfig,
    SexPopulationConfig,
)

# Step 1: Setup Fixtures


@pytest.fixture
def sex_data_config():
    return SexDataConfig(
        index=["age_group_id", "year_id", "location_id"],
        val="val",
        val_sd="val_sd",
    )


@pytest.fixture
def sex_pattern_config():
    return SexPatternConfig(by=["age_group_id", "year_id"])


@pytest.fixture
def sex_population_config():
    return SexPopulationConfig(
        index=["age_group_id", "year_id", "location_id"],
        sex="sex_id",
        sex_m=1,
        sex_f=2,
        val="population",
    )


@pytest.fixture
def valid_data():
    return pd.DataFrame(
        {
            "age_group_id": [1, 1, 2, 2],
            "year_id": [2000, 2000, 2001, 2001],
            "location_id": [10, 20, 10, 20],
            "sex_id": [3, 3, 3, 3],
            "val": [100, 200, 150, 250],
            "val_sd": [10, 20, 15, 25],
        }
    )


@pytest.fixture
def sex_splitter(sex_data_config, sex_pattern_config, sex_population_config):
    return SexSplitter(
        data=sex_data_config,
        pattern=sex_pattern_config,
        population=sex_population_config,
    )


# Step 2: Write Tests for parse_data


def test_parse_data_missing_columns(sex_splitter, valid_data):
    """Test parse_data raises an error when columns are missing."""
    invalid_data = valid_data.drop(columns=["val"])
    with pytest.raises(KeyError, match="Missing columns"):
        sex_splitter.parse_data(invalid_data)


def test_parse_data_duplicated_index(sex_splitter, valid_data):
    """Test parse_data raises an error on duplicated index."""
    duplicated_data = pd.concat([valid_data, valid_data])
    with pytest.raises(ValueError, match="Duplicated index found"):
        sex_splitter.parse_data(duplicated_data)


def test_parse_data_with_nan(sex_splitter, valid_data):
    """Test parse_data raises an error when there are NaN values."""
    nan_data = valid_data.copy()
    nan_data.loc[0, "val"] = None
    with pytest.raises(ValueError, match="NaN values found"):
        sex_splitter.parse_data(nan_data)


def test_parse_data_non_positive(sex_splitter, valid_data):
    """Test parse_data raises an error for non-positive values in val or val_sd."""
    non_positive_data = valid_data.copy()
    non_positive_data.loc[0, "val"] = -10
    with pytest.raises(ValueError, match="Non-positive values found"):
        sex_splitter.parse_data(non_positive_data)


def test_parse_data_valid(sex_splitter, valid_data):
    """Test that parse_data works correctly on valid data."""
    parsed_data = sex_splitter.parse_data(valid_data)
    assert not parsed_data.empty
    assert "val" in parsed_data.columns
    assert "val_sd" in parsed_data.columns


def test_parse_data_invalid_sex_rows(sex_splitter, valid_data):
    """Test parse_data raises an error if invalid sex_id rows are present."""
    invalid_sex_data = valid_data.copy()
    invalid_sex_data.loc[0, "sex_id"] = 1  # Setting sex_id to sex_m
    with pytest.raises(ValueError, match="Invalid rows"):
        sex_splitter.parse_data(invalid_sex_data)
