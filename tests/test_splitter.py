import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError
from pydisagg.ihme.splitter import (
    AgeSplitter,
    DataConfig,
    PatternConfig,
    PopulationConfig,
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
            location_id=[1]*10,
        )
    )
    pattern_df2 = pattern_df1.copy()
    pattern_df2['location_id']=2
    return pd.concat([pattern_df1,pattern_df2]).reset_index(drop=True)

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


def test_model_post_init():
    data_config = DataConfig(
        index=["index1"],
        age_lwr="age_lwr1",
        age_upr="age_upr1",
        val="val1",
        val_sd="val_sd1",
    )
    pattern_config = PatternConfig(
        by=["by1"],
        age_key="age_key1",
        age_lwr="age_lwr2",
        age_upr="age_upr2",
        draws=["draw1"],
        draw_mean="draw_mean1",
        draw_var="draw_var1",
        prefix="prefix1",
    )
    population_config = PopulationConfig(
        index=["index2"], val="val2", prefix="prefix2"
    )

    # Test when pattern.by is not a subset of data.index
    with pytest.raises(ValidationError):
        AgeSplitter(
            data=data_config,
            pattern=pattern_config,
            population=population_config,
        )

    # Test when pattern.age_key is in data.index
    data_config.index.append("age_key1")
    pattern_config.by.append("index1")
    with pytest.raises(ValidationError):
        AgeSplitter(
            data=data_config,
            pattern=pattern_config,
            population=population_config,
        )

    # Test when pattern.age_key is not in population.index
    data_config.index.remove("age_key1")
    population_config.index.remove("index2")
    with pytest.raises(ValidationError):
        AgeSplitter(
            data=data_config,
            pattern=pattern_config,
            population=population_config,
        )

    # Test when population.index is not a subset of data.index + pattern.index
    population_config.index.append("index3")
    with pytest.raises(ValidationError):
        AgeSplitter(
            data=data_config,
            pattern=pattern_config,
            population=population_config,
        )


@pytest.mark.skip(reason="not implemented yet")
def test_parse_data(data):
    data_config = DataConfig(
        index=["unique_id", "location_id", "year_id", "sex_id"],
        age_lwr="age_start",
        age_upr="age_end",
        val="mean",
        val_sd="SE",
    )
    splitter = AgeSplitter(data=data_config, pattern=None, population=None)
    parsed_data = splitter.parse_data(data)
    assert isinstance(parsed_data, pd.DataFrame)
    assert not parsed_data.empty
    assert set(data_config.index).issubset(parsed_data.columns)
    assert data_config.age_lwr in parsed_data.columns
    assert data_config.age_upr in parsed_data.columns
    assert data_config.val in parsed_data.columns
    assert data_config.val_sd in parsed_data.columns


@pytest.mark.skip(reason="not implemented yet")
def test_parse_pattern(pattern):
    splitter = AgeSplitter()
    # Add assertions for each test case


@pytest.mark.skip(reason="not implemented yet")
def test_merge_with_pattern(data, pattern):
    splitter = AgeSplitter()
    # Add assertions for each test case


@pytest.mark.skip(reason="not implemented yet")
def test_parse_population(population):
    splitter = AgeSplitter()
    # Add assertions for each test case


@pytest.mark.skip(reason="not implemented yet")
def test_merge_with_population(data, population):
    splitter = AgeSplitter()
    # Add assertions for each test case


@pytest.mark.skip(reason="not implemented yet")
def test_align_pattern_and_population(data):
    splitter = AgeSplitter()
    # Add assertions for each test case


@pytest.mark.skip(reason="not implemented yet")
def test_split(data, pattern, population):
    splitter = AgeSplitter()
    # Add assertions for each test case
