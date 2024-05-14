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

    draw_mean: str = "avg_draw"
    draw_var: str = "var_draw"
    prefix: str = "pat_"

    @property
    def index(self) -> list[str]:
        return self.by + [self.age_key]

    @property
    def columns(self) -> list[str]:
        return self.index + [
            self.age_lwr,
            self.age_upr,
            self.draw_mean,
            self.draw_var,
        ]

    @property
    def val_fields(self) -> list[str]:
        return [
            "age_lwr",
            "age_upr",
            "draw_mean",
            "draw_var",
        ]

    def apply_prefix(self) -> dict[str, str]:
        rename_map = {}
        for field in self.val_fields:
            field_val = getattr(self, field)
            new_field_val = self.prefix + field_val
            setattr(self, field, new_field_val)
            rename_map[field_val] = new_field_val
        return rename_map


class PopulationConfig(BaseModel):
    index: list[str]
    val: str

    prefix: str = "pop_"

    @property
    def columns(self) -> list[str]:
        return self.index + [self.val]

    @property
    def val_fields(self) -> list[str]:
        return ["val"]

    def apply_prefix(self) -> dict[str, str]:
        rename_map = {}
        for field in self.val_fields:
            field_val = getattr(self, field)
            new_field_val = self.prefix + field_val
            setattr(self, field, new_field_val)
            rename_map[field_val] = new_field_val
        return rename_map


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
        validate_columns(pattern, self.pattern.draws, name)
        pattern[self.pattern.draw_mean] = pattern[self.pattern.draws].mean(
            axis=1
        )
        pattern[self.pattern.draw_var] = pattern[self.pattern.draws].var(axis=1)

        validate_columns(pattern, self.pattern.columns, name)
        pattern = pattern[self.pattern.columns].copy()

        rename_map = self.pattern.apply_prefix()
        pattern.rename(columns=rename_map, inplace=True)

        validate_index(pattern, self.pattern.index, name)
        validate_nonan(pattern, name)
        validate_interval(
            pattern,
            self.pattern.age_lwr,
            self.pattern.age_upr,
            self.pattern.index,
            name,
        )

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
        data_with_pattern = (
            data.merge(pattern, on=self.pattern.by, how="left")
            .query(
                f"({self.pattern.age_lwr} >= {self.data.age_lwr} and"
                f" {self.pattern.age_lwr} < {self.data.age_upr}) or"
                f"({self.pattern.age_upr} > {self.data.age_lwr} and"
                f" {self.pattern.age_upr} <= {self.data.age_upr})"
            )
            .dropna()
        )
        return data_with_pattern

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "population"
        validate_columns(population, self.population.columns, name)

        population = population[self.population.columns].copy()
        rename_map = self.population.apply_prefix()
        population.rename(columns=rename_map, inplace=True)

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
        ).dropna()
        return data_with_population

    def _align_pattern_and_population(self, data: DataFrame) -> DataFrame:
        warn("This is only naive constant interpolation.")

        data = data.sort_values(
            self.data.index + [self.pattern.age_lwr, self.pattern.age_upr],
            ignore_index=True,
        )

        data[self.pattern.draw_mean + "_aligned"] = data[self.pattern.draw_mean]
        data[self.pattern.draw_var + "_aligned"] = data[self.pattern.draw_var]
        data[self.population.val + "_aligned"] = data[self.population.val]

        data_group = data.groupby(self.data.index, sort=False)
        for key, data_sub in data_group:
            # first element of the group
            index_first, index_last = data_group.groups[key][[0, -1]]
            data_first, data_last = data.loc[index_first], data.loc[index_last]

            # align pattern
            # TODO: currently we do constant interpolation for pattern
            data.loc[index_first, self.pattern.draw_mean + "_aligned"] = (
                data_first[self.pattern.draw_mean]
            )
            data.loc[index_first, self.pattern.draw_var + "_aligned"] = (
                data_first[self.pattern.draw_var]
            )
            data.loc[index_last, self.pattern.draw_mean + "_aligned"] = (
                data_last[self.pattern.draw_mean]
            )
            data.loc[index_last, self.pattern.draw_var + "_aligned"] = (
                data_last[self.pattern.draw_var]
            )

            # align population
            # TODO: this is naive implementation compute the proprotion of population
            # within the given age interval
            data.loc[index_first, self.population.val + "_aligned"] = (
                data_first[self.population.val]
                / (
                    data_first[self.pattern.age_upr]
                    - data_first[self.pattern.age_lwr]
                )
                * (
                    data_first[self.pattern.age_upr]
                    - data_first[self.data.age_lwr]
                )
            )
            data.loc[index_last, self.population.val + "_aligned"] = (
                data_last[self.population.val]
                / (
                    data_last[self.pattern.age_upr]
                    - data_last[self.pattern.age_lwr]
                )
                * (
                    data_last[self.data.age_upr]
                    - data_last[self.pattern.age_lwr]
                )
            )

        return data

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: DisaggModel = RateMultiplicativeModel(),
        output_type: str = "rate",
    ) -> DataFrame:
        """
        Splits the data based on the given pattern and population. The split results are added to the data as new columns.

        Parameters
        ----------
        data : DataFrame
            The data to be split.
        pattern : DataFrame
            The pattern to be used for splitting the data.
        population : DataFrame
            The population to be used for splitting the data.
        model : DisaggModel, optional
            The model to be used for splitting the data, by default RateMultiplicativeModel().
        output_type : str, optional
            The type of output to be returned, by default "rate".

        Returns
        -------
        DataFrame
            The two main output columns are: 'split_result' and 'split_result_se'.
            There are additional intermediate columns for sanity checks and calculations (have a prefix of pat_ or pop_, and a suffix of _aligned).

        """
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
                bucket_populations=data_sub[
                    self.population.val + "_aligned"
                ].to_numpy(),
                rate_pattern=data_sub[
                    self.pattern.draw_mean + "_aligned"
                ].to_numpy(),
                model=model,
                output_type=output_type,
                normalize_pop_for_average_type_obs=True,
                observed_total_se=data_sub[self.data.val_sd].iloc[0],
                pattern_covariance=np.diag(
                    data_sub[self.pattern.draw_var + "_aligned"].to_numpy()
                ),
            )
            index = data_group.groups[key]
            data.loc[index, "split_result"] = split_result
            data.loc[index, "split_result_se"] = SE

        return data
