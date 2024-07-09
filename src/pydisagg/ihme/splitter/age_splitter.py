import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_interval,
    validate_noindexdiff,
    validate_nonan,
    validate_pat_coverage,
    validate_positive,
)
from pydisagg.models import LogOdds_model, RateMultiplicativeModel


class AgeDataConfig(BaseModel):
    index: list[str]
    age_lwr: str
    age_upr: str
    val: str
    val_sd: str
    # sample_size: str | None

    @property
    def columns(self) -> list[str]:
        base_columns = self.index + [
            self.age_lwr,
            self.age_upr,
            self.val,
            self.val_sd,
        ]
        # if self.sample_size is not None:
        #     base_columns.append(self.sample_size)
        return list(set(base_columns))


class AgePopulationConfig(BaseModel):
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
            new_field_val = self.prefix + (field_val := getattr(self, field))
            rename_map[field_val] = new_field_val
            setattr(self, field, new_field_val)
        return rename_map

    def remove_prefix(self) -> None:
        for field in self.val_fields:
            setattr(self, field, getattr(self, field).removeprefix(self.prefix))


class AgePatternConfig(BaseModel):
    by: list[str]
    age_key: str
    age_lwr: str
    age_upr: str

    draws: list[str] = []
    val: str = "val"
    val_sd: str = "val_sd"
    prefix: str = "pat_"

    @property
    def index(self) -> list[str]:
        return self.by + [self.age_key]

    @property
    def columns(self) -> list[str]:
        return self.index + [
            self.age_lwr,
            self.age_upr,
            self.val,
            self.val_sd,
        ]

    @property
    def val_fields(self) -> list[str]:
        return [
            "age_lwr",
            "age_upr",
            "val",
            "val_sd",
        ]

    def apply_prefix(self) -> dict[str, str]:
        rename_map = {}
        for field in self.val_fields:
            new_field_val = self.prefix + (field_val := getattr(self, field))
            rename_map[field_val] = new_field_val
            setattr(self, field, new_field_val)
        return rename_map

    def remove_prefix(self) -> None:
        for field in self.val_fields:
            setattr(self, field, getattr(self, field).removeprefix(self.prefix))


class AgeSplitter(BaseModel):
    data: AgeDataConfig
    pattern: AgePatternConfig
    population: AgePopulationConfig

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

    def parse_data(self, data: DataFrame, positive_strict: bool) -> DataFrame:
        name = "Parsing Data"
        validate_columns(data, self.data.columns, name)

        data = data[self.data.columns].copy()

        validate_index(data, self.data.index, name)
        validate_nonan(data, name)
        validate_positive(
            data, [self.data.val_sd], name, strict=positive_strict
        )
        validate_interval(
            data, self.data.age_lwr, self.data.age_upr, self.data.index, name
        )
        return data

    def parse_pattern(
        self, data: DataFrame, pattern: DataFrame, positive_strict: bool
    ) -> DataFrame:
        name = "Parsing Pattern"

        if not all(
            col in pattern.columns
            for col in [self.pattern.val, self.pattern.val_sd]
        ):
            if not self.pattern.draws:
                raise ValueError(
                    "Must provide draws for pattern if pattern.val and "
                    "pattern.val_sd are not available."
                )

            validate_columns(pattern, self.pattern.draws, name)
            pattern[self.pattern.val] = pattern[self.pattern.draws].mean(axis=1)
            pattern[self.pattern.val_sd] = pattern[self.pattern.draws].std(
                axis=1
            )

        validate_columns(pattern, self.pattern.columns, name)
        pattern = pattern[self.pattern.columns].copy()

        validate_index(pattern, self.pattern.index, name)
        validate_nonan(pattern, name)
        validate_positive(
            pattern, [self.pattern.val_sd], name, strict=positive_strict
        )
        validate_interval(
            pattern,
            self.pattern.age_lwr,
            self.pattern.age_upr,
            self.pattern.index,
            name,
        )

        pattern_copy = pattern.copy()
        rename_map = self.pattern.apply_prefix()
        pattern_copy.rename(columns=rename_map, inplace=True)

        data_with_pattern = self._merge_with_pattern(data, pattern_copy)

        validate_noindexdiff(data, data_with_pattern, self.data.index, name)
        validate_pat_coverage(
            data_with_pattern,
            self.data.age_lwr,
            self.data.age_upr,
            self.pattern.age_lwr,
            self.pattern.age_upr,
            self.data.index,
            name,
        )
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
                f" {self.pattern.age_upr} <= {self.data.age_upr}) or"
                f"({self.pattern.age_lwr} <= {self.data.age_lwr} and"
                f" {self.pattern.age_upr} >= {self.data.age_upr})"
            )
            .dropna()
        )
        return data_with_pattern

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "Parsing Population"
        validate_columns(population, self.population.columns, name)

        population = population[self.population.columns].copy()

        validate_index(population, self.population.index, name)
        validate_nonan(population, name)

        pop_copy = population.copy()
        rename_map = self.population.apply_prefix()
        pop_copy.rename(columns=rename_map, inplace=True)

        data_with_population = self._merge_with_population(data, pop_copy)

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
        data = data.sort_values(
            self.data.index + [self.pattern.age_lwr, self.pattern.age_upr],
            ignore_index=True,
        )

        data[self.pattern.val + "_aligned"] = data[self.pattern.val]
        data[self.pattern.val_sd + "_aligned"] = data[self.pattern.val_sd]
        data[self.population.val + "_aligned"] = data[self.population.val]

        index_group = data.reset_index().groupby(self.data.index)["index"]
        index_first = index_group.first().to_list()
        index_last = index_group.last().to_list()

        data.loc[index_first, self.population.val + "_aligned"] = data.loc[
            index_first
        ].eval(
            f"{self.population.val} "
            f"/ ({self.pattern.age_upr} - {self.pattern.age_lwr}) "
            f"* ({self.pattern.age_upr} - {self.data.age_lwr})"
        )

        data.loc[index_last, self.population.val + "_aligned"] = data.loc[
            index_last
        ].eval(
            f"{self.population.val} "
            f"/ ({self.pattern.age_upr} - {self.pattern.age_lwr}) "
            f"* ({self.data.age_upr} - {self.pattern.age_lwr})"
        )

        # Not used right now, but useful in checking how we handle population partitioning
        # Can be used to split sample sizes using the pseudo-proportion
        data[self.population.val + "_total"] = data.groupby(self.data.index)[
            self.population.val + "_aligned"
        ].transform(lambda x: x.sum())
        data[self.population.val + "_proportion"] = (
            data[self.population.val + "_aligned"]
            / data[self.population.val + "_total"]
        )

        return data

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: Literal["rate", "logodds"] = "rate",
        output_type: Literal["rate", "count"] = "rate",
        propagate_zeros=False,
    ) -> DataFrame:
        """
        Splits the data based on the given pattern and population. The split
        results are added to the data as new columns.

        Parameters
        ----------
        data : DataFrame
            The data to be split.
        pattern : DataFrame
            The pattern to be used for splitting the data.
        population : DataFrame
            The population to be used for splitting the data.
        model : str, optional
            The model to be used for splitting the data, by default "rate".
            Can be "rate" or "logodds".
        output_type : str, optional
            The type of output to be returned, by default "rate".
        propagate_zeros : Bool, optional
            Whether to propagate pre-split zeros as post split zeros. Default false

        Returns
        -------
        DataFrame
            The two main output columns are: 'age_split_result' and 'age_split_result_se'.
            There are additional intermediate columns for sanity checks and
            calculations (have a prefix of pat_ or pop_, and a suffix of _aligned).

        """
        model_mapping = {
            "rate": RateMultiplicativeModel(),
            "logodds": LogOdds_model(),
        }

        if model not in model_mapping:
            raise ValueError(
                f"Invalid model: {model}. Expected one of: {list(model_mapping.keys())}"
            )

        model_instance = model_mapping[model]

        # If not propagating zeros,then positivity has to be strict
        data = self.parse_data(data, positive_strict=not propagate_zeros)
        data = self.parse_pattern(
            data, pattern, positive_strict=not propagate_zeros
        )
        data = self.parse_population(data, population)

        data = self._align_pattern_and_population(data)

        # where split happens
        data["age_split_result"], data["age_split_result_se"] = np.nan, np.nan
        if propagate_zeros is True:
            data_zero = data[
                (data[self.data.val] == 0)
                | (data[self.pattern.val + "_aligned"] == 0)
            ]
            data = data[data[self.data.val] > 0]
            # Manually split zero values
            data_zero["age_split_result"] = 0.0
            data_zero["age_split_result_se"] = 0.0

            # Warn for all zero propagation
            num_zval = (data[self.data.val] == 0).sum()
            num_zpat = (data[self.pattern.val + "_aligned"] == 0).sum()
            num_overlap = (
                (data[self.data.val] == 0)
                * (data[self.pattern.val + "_aligned"] == 0)
            ).sum()
            if num_zval > 0:
                warnings.warn(
                    f"{num_zval} zeros produced from propagating pre-split zero values"
                )
            if num_zpat > 0:
                warnings.warn(
                    f"{num_zpat} zeros produced from propagating pattern zero values"
                )
            if num_overlap > 0:
                warnings.warn(
                    f"{num_overlap} zeros produced from this were overlappingf"
                )

        data_group = data.groupby(self.data.index)
        if output_type == "count":
            pop_normalize = False
        elif output_type == "rate":
            pop_normalize = True

        for key, data_sub in data_group:
            split_result, SE = split_datapoint(
                observed_total=data_sub[self.data.val].iloc[0],
                bucket_populations=data_sub[
                    self.population.val + "_aligned"
                ].to_numpy(),
                rate_pattern=data_sub[self.pattern.val + "_aligned"].to_numpy(),
                model=model_instance,
                output_type=output_type,  # type: ignore, this is handeled by model_mapping
                normalize_pop_for_average_type_obs=pop_normalize,
                observed_total_se=data_sub[self.data.val_sd].iloc[0],
                pattern_covariance=np.diag(
                    data_sub[self.pattern.val_sd + "_aligned"].to_numpy() ** 2
                ),
            )
            index = data_group.groups[key]
            data.loc[index, "age_split_result"] = split_result
            data.loc[index, "age_split_result_se"] = SE
        if propagate_zeros is True:
            data = pd.concat([data, data_zero])

        self.pattern.remove_prefix()
        self.population.remove_prefix()

        # Something like this can be implemented for sample size split
        # data["split_"+ self.data.sample_size] = data[self.data.sample_size] * data[self.population.val + "_proportion"]

        return data
