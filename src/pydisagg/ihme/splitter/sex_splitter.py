from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.schema import Schema
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_noindexdiff,
    validate_nonan,
    validate_positive,
)
from pydisagg.models import RateMultiplicativeModel


class SexPatternConfig(Schema):
    by: list[str]
    draws: list[str] = []
    val: str = "ratio_f_to_m"
    val_sd: str = "ratio_f_to_m_se"
    prefix: str = "sex_pat_"

    @property
    def index(self) -> list[str]:
        return self.by

    @property
    def columns(self) -> list[str]:
        return self.index + [
            self.val,
            self.val_sd,
        ]

    @property
    def val_fields(self) -> list[str]:
        return [
            "val",
            "val_sd",
        ]


class SexPopulationConfig(Schema):
    index: list[str]
    sex: str
    sex_m: str | int
    sex_f: str | int
    val: str

    @property
    def columns(self) -> list[str]:
        return self.index + [self.sex, self.val]

    @property
    def val_fields(self) -> list[str]:
        return ["val"]


class SexDataConfig(Schema):
    index: list[str]
    val: str
    val_sd: str

    @property
    def columns(self) -> list[str]:
        return list(set(self.index + [self.val, self.val_sd]))


class SexSplitter(BaseModel):
    data: SexDataConfig
    pattern: SexPatternConfig
    population: SexPopulationConfig

    def model_post_init(self, __context: Any) -> None:
        """Extra validation of all the index."""
        if not set(self.pattern.by).issubset(self.data.index):
            raise ValueError("pattern.by must be a subset of data.index")
        if not set(self.population.index).issubset(
            self.data.index + self.pattern.index
        ):
            raise ValueError(
                "population.index must be a subset of data.index + pattern.index"
            )

    def _merge_with_pattern(
        self, data: DataFrame, pattern: DataFrame
    ) -> DataFrame:
        data_with_pattern = data.merge(
            pattern, on=self.pattern.by, how="left"
        ).dropna()
        return data_with_pattern

    def parse_data(self, data: DataFrame) -> DataFrame:
        name = "When parsing, data"
        validate_columns(data, self.data.columns, name)
        data = data[self.data.columns].copy()
        validate_index(data, self.data.index, name)
        validate_nonan(data, name)
        validate_positive(data, [self.data.val_sd], name)
        return data

    def parse_pattern(self, data: DataFrame, pattern: DataFrame) -> DataFrame:
        name = "When parsing, pattern"
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
        validate_positive(pattern, [self.pattern.val_sd], name, strict=False)
        data_with_pattern = self._merge_with_pattern(data, pattern)
        return data_with_pattern

    def get_population_by_sex(self, population, sex_value):
        return population[population[self.population.sex] == sex_value][
            self.population.index + [self.population.val]
        ].copy()

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "When parsing, population"
        validate_columns(population, self.population.columns, name)

        male_population = self.get_population_by_sex(
            population, self.population.sex_m
        )

        female_population = self.get_population_by_sex(
            population, self.population.sex_f
        )

        male_population.rename(
            columns={self.population.val: "m_pop"}, inplace=True
        )
        female_population.rename(
            columns={self.population.val: "f_pop"}, inplace=True
        )

        data_with_population = self._merge_with_population(
            data, male_population, "m_pop"
        )

        data_with_population = self._merge_with_population(
            data_with_population, female_population, "f_pop"
        )

        validate_columns(data_with_population, ["m_pop", "f_pop"], name)
        validate_nonan(data_with_population, name)
        validate_noindexdiff(data, data_with_population, self.data.index, name)
        return data_with_population

    def _merge_with_population(
        self, data: DataFrame, population: DataFrame, pop_col: str
    ) -> DataFrame:
        keep_cols = self.population.index + [pop_col]
        population_temp = population[keep_cols]
        data_with_population = data.merge(
            population_temp,
            on=self.population.index,
            how="left",
        )
        return data_with_population

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: str = "rate",
        output_type: str = "rate",
    ) -> DataFrame:
        data = self.parse_data(data)
        data = self.parse_pattern(data, pattern)
        data = self.parse_population(data, population)

        if output_type == "count":
            pop_normalize = False
        elif output_type == "rate":
            pop_normalize = True

        def sex_split_row(row):
            split_result, SE = split_datapoint(
                # This comes from the data
                observed_total=row[self.data.val],
                bucket_populations=np.array([row["m_pop"], row["f_pop"]]),
                # This is from sex_pattern
                rate_pattern=np.array([1.0, row[self.pattern.val]]),
                model=RateMultiplicativeModel(),
                output_type=output_type,
                normalize_pop_for_average_type_obs=pop_normalize,
                # This is from the data
                observed_total_se=row[self.data.val_sd],
                # This is from sex_pattern
                pattern_covariance=np.diag(
                    np.array([0, row[self.pattern.val_sd] ** 2])
                ),
            )
            return pd.Series(
                {
                    "split_val_male": split_result[0],
                    "split_val_female": split_result[1],
                    "se_male": SE[0],
                    "se_female": SE[1],
                }
            )

        split_results = data.apply(sex_split_row, axis=1)
        split_df_male = data.copy()
        split_df_female = data.copy()

        split_df_male["sex_split_result"] = split_results["split_val_male"]
        split_df_male["sex_split_result_se"] = split_results["se_male"]
        split_df_female["sex_split_result"] = split_results["split_val_female"]
        split_df_female["sex_split_result_se"] = split_results["se_female"]

        split_df_male["sex_id"] = self.population.sex_m
        split_df_female["sex_id"] = self.population.sex_f

        final_split_df = (
            pd.concat([split_df_male, split_df_female], ignore_index=True)
            .sort_values(self.data.index)
            .reset_index(drop=True)
        )
        final_split_df = final_split_df.reindex(
            columns=self.data.index
            + [
                col
                for col in final_split_df.columns
                if col not in self.data.index
            ]
        )
        return final_split_df
