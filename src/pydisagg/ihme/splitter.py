from typing import Any
from warnings import warn

import numpy as np
from pandas import DataFrame
from pydantic import BaseModel

from pydisagg.DisaggModel import DisaggModel
from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_interval,
    validate_noindexdiff,
    validate_nonan,
    validate_positive,
)
from pydisagg.models import RateMultiplicativeModel


class DataConfig(BaseModel):
    index: list[str]
    age_lwr: str
    age_upr: str
    val: str
    val_sd: str

    @property
    def columns(self) -> list[str]:
        return self.index + [self.age_lwr, self.age_upr, self.val, self.val_sd]


class PatternConfig(BaseModel):
    by: list[str]
    age_key: str
    age_lwr: str
    age_upr: str
    draws: list[str]

    @property
    def index(self) -> list[str]:
        return self.by + [self.age_key]

    @property
    def columns(self) -> list[str]:
        return self.index + [
            self.age_lwr,
            self.age_upr,
        ]


class PopulationConfig(BaseModel):
    index: list[str]
    val: str

    @property
    def columns(self) -> list[str]:
        return self.index + [self.val]


class AgeSplitter(BaseModel):
    data: DataConfig
    pattern: PatternConfig
    population: PopulationConfig

    def model_post_init(self, __context: Any) -> None:
        """Extra validation of all the index."""
        if not set(self.pattern.by).issubset(self.data.index):
            raise ValueError("pattern.by must be a subset of data.index")
        if self.pattern.age_key in self.data.index:
            raise ValueError("pattern.age_key must not be in data.index")
        if self.pattern.age_key not in self.population.index:
            raise ValueError("pattern.age_key must be in population.index")
        if not set(self.population.index).issubset(
            self.data.index + self.pattern.index
        ):
            raise ValueError(
                "population.index must be a subset of data.index + pattern.index"
            )

    def parse_data(self, data: DataFrame) -> DataFrame:
        name = "data"
        validate_columns(data, self.data.columns, name)

        data = data[self.data.columns].copy()

        validate_index(data, self.data.index, name)
        validate_nonan(data, name)
        validate_positive(data, [self.data.val_sd], name)
        validate_interval(
            data, self.data.age_lwr, self.data.age_upr, self.data.index, name
        )
        return data

    def parse_pattern(self, data: DataFrame, pattern: DataFrame) -> DataFrame:
        name = "pattern"
        # Currently working around validate columns because mean_draw and mean_var are going to be created
        validate_columns(pattern, self.pattern.columns, name)

        columns_with_draws = list(self.pattern.columns) + list(
            self.pattern.draws
        )
        pattern = pattern[columns_with_draws].copy()
        # pattern = pattern[self.pattern.columns].copy()

        validate_index(pattern, self.pattern.index, name)
        validate_nonan(pattern, name)
        validate_interval(
            pattern,
            self.pattern.age_lwr,
            self.pattern.age_upr,
            self.pattern.index,
            name,
        )

        pattern["avg_draw"] = pattern[self.pattern.draws].mean(axis=1)
        pattern["var_draw"] = pattern[self.pattern.draws].var(axis=1)
        pattern = pattern.drop(columns=self.pattern.draws)

        data_with_pattern = self._merge_with_pattern(data, pattern)

        validate_noindexdiff(data, data_with_pattern, self.data.index, name)
        # TODO: add validation checks for incomplete age pattern
        # * pattern age intervals do not overlap
        # * smallest pattern interval doesn't cover the left end point of data
        # * largest pattern interval doesn't cover the right end point of data
        # How to vectorize this action...
        return data_with_pattern

    def _merge_with_pattern(
        self, data: DataFrame, pattern: DataFrame
    ) -> DataFrame:
        data_with_pattern = data.merge(
            pattern, on=self.pattern.by, how="left", suffixes=("", "_pat")
        )

        # Removed suffix from query because there was no name overlap, and also they are called from each infividual df not the combined so the rename wouldn't even apply yet I dont think
        data_with_pattern = data_with_pattern.query(
            f"({self.pattern.age_lwr}_pat >= {self.data.age_lwr} and"
            f" {self.pattern.age_lwr}_pat < {self.data.age_upr}) or"
            f"({self.pattern.age_upr}_pat > {self.data.age_lwr} and"
            f" {self.pattern.age_upr}_pat <= {self.data.age_upr})"
        ).dropna()
        return data_with_pattern

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "population"
        validate_columns(population, self.population.columns, name)

        population = population[self.population.columns].copy()

        validate_index(population, self.population.index, name)
        validate_nonan(population, name)

        data_with_population = self._merge_with_population(data, population)

        validate_noindexdiff(
            data,
            data_with_population,
            self.data.index + [self.pattern.age_key],
            name,
        )
        return data_with_population

    def _merge_with_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        data_with_population = data.merge(
            population,
            on=self.population.index,
            how="left",
            suffixes=("", "_pop"),
        ).dropna()
        return data_with_population

    def _align_pattern_and_population(self, data: DataFrame) -> DataFrame:
        warn("Not implemented yet")
        return data

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: DisaggModel = RateMultiplicativeModel(),
        output_type: str = "rate",
    ) -> DataFrame:
        data = self.parse_data(data)
        data = self.parse_pattern(data, pattern)
        data = self.parse_population(data, population)

        data = self._align_pattern_and_population(data)

        # where split happen
        data["split_result"], data["split_result_se"] = np.nan, np.nan
        data_group = data.groupby(self.data.index)
        for key, data_sub in data_group:
            split_result, SE = split_datapoint(
                observed_total=data_sub[self.data.val].iloc[0],
                bucket_populations=data_sub[self.population.val].to_numpy(),
                rate_pattern=data_sub["avg_draw"].to_numpy(),
                model=model,
                output_type=output_type,
                normalize_pop_for_average_type_obs=True,
                observed_total_se=data_sub[self.data.val_sd].iloc[0],
                pattern_covariance=np.diag(data_sub["var_draw"].to_numpy()),
            )
            index = data_group.groups[key]
            data.loc[index, "split_result"] = split_result
            data.loc[index, "split_result_se"] = SE

        return data
