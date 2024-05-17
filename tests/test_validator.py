import pytest
import pandas as pd
import numpy as np
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_nonan,
    validate_positive,
    validate_interval,
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
    return pd.DataFrame(
        dict(
            sex_id=[1] * 5 + [2] * 5,
            age_start=[0, 5, 10, 15, 20] * 2,
            age_end=[5, 10, 15, 20, 25] * 2,
            age_group_id=list(range(5)) * 2,
            draw_0=np.random.rand(10),
            draw_1=np.random.rand(10),
            draw_2=np.random.rand(10),
            year_id=[2010] * 10,
            location_id=[1, 2] * 5,
        )
    )


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


def test_validate_columns_missing(population):
    with pytest.raises(KeyError):
        validate_columns(
            population.drop(columns=["population"]),
            ["sex_id", "location_id", "age_group_id", "year_id", "population"],
            "population",
        )


def test_validate_index_missing(population):
    with pytest.raises(ValueError):
        validate_index(
            pd.concat([population, population]),
            ["sex_id", "location_id", "age_group_id", "year_id"],
            "population",
        )


def test_validate_nonan(population):
    with pytest.raises(ValueError):
        validate_nonan(population.assign(population=np.nan), "population")


def test_validate_positive_strict(population):
    with pytest.raises(ValueError):
        validate_positive(
            population.assign(population=0),
            ["population"],
            "population",
            strict=True,
        )


def test_validate_positive_not_strict(population):
    with pytest.raises(ValueError):
        validate_positive(
            population.assign(population=-1),
            ["population"],
            "population",
            strict=False,
        )


def test_validate_interval_lower_equal_upper(data):
    with pytest.raises(ValueError):
        validate_interval(
            data.assign(age_end=data["age_start"]),
            "age_start",
            "age_end",
            ["uid"],
            "data",
        )


def test_validate_interval_lower_greater_than_upper(data):
    with pytest.raises(ValueError):
        validate_interval(
            data.assign(age_end=0), "age_start", "age_end", ["uid"], "data"
        )


def test_validate_interval_positive(data):
    validate_interval(data, "age_start", "age_end", ["uid"], "data")


def test_validate_positive_no_error(population):
    validate_positive(population, ["population"], "population", strict=True)
    validate_positive(population, ["population"], "population", strict=False)


@pytest.fixture
def merged_data_pattern(data, pattern):
    return pd.merge(
        pattern,
        data,
        on=["sex_id", "age_start", "age_end", "location_id"],
        how="outer",
    )


def test_validate_noindexdiff_merged_positive(merged_data_pattern, population):
    # Positive test case: no index difference
    validate_noindexdiff(
        population,
        merged_data_pattern,
        ["sex_id", "location_id"],
        "merged_data_pattern",
    )


def test_validate_noindexdiff_merged_negative(data, pattern):
    # Negative test case: index difference exists
    data_with_pattern = data.merge(
        pattern,
        on=["sex_id", "age_start", "age_end", "location_id", "year_id"],
        how="left",
    )
    # Introduce an index difference
    data_with_pattern = data_with_pattern.drop(index=0)
    with pytest.raises(ValueError):
        validate_noindexdiff(
            data,
            data_with_pattern,
            data.columns.tolist(),
            "merged_data_pattern",
        )
