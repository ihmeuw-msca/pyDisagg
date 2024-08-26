from typing import Any
import pandas as pd
import numpy as np
from pandas import DataFrame
from pydantic import BaseModel
from scipy.special import expit
from typing import Literal
from pydisagg.disaggregate import split_datapoint
from pydisagg.models import RateMultiplicativeModel, LogOdds_model
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_nonan,
    validate_positive,
    validate_noindexdiff,
    validate_realnumber,
)


class SexPatternConfig(BaseModel):
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


class SexPopulationConfig(BaseModel):
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


class SexDataConfig(BaseModel):
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

    def get_population_by_sex(self, population, sex_value):
        return population[population[self.population.sex] == sex_value][
            self.population.index + [self.population.val]
        ].copy()

    def parse_data(self, data: DataFrame) -> DataFrame:
        name = "While parsing data"

        # Validate core columns first
        try:
            validate_columns(data, self.data.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the input data. Details:\n{e}"
            )

        if self.population.sex not in data.columns:
            raise KeyError(
                f"{name}: Missing column '{self.population.sex}' in the input data."
            )

        try:
            validate_index(data, self.data.index, name)
        except ValueError as e:
            raise ValueError(f"{name}: Duplicated index found. Details:\n{e}")

        try:
            validate_nonan(data, name)
        except ValueError as e:
            raise ValueError(f"{name}: NaN values found. Details:\n{e}")

        try:
            validate_positive(data, [self.data.val, self.data.val_sd], name)
        except ValueError as e:
            raise ValueError(
                f"{name}: Non-positive values found in 'val' or 'val_sd'. Details:\n{e}"
            )

        # Validate that no rows have sex_id equal to sex_m or sex_f
        invalid_sex_rows = data[
            data[self.population.sex].isin(
                [self.population.sex_m, self.population.sex_f]
            )
        ]
        if not invalid_sex_rows.empty:
            raise ValueError(
                f"{name}: The input data contains rows where the '{self.population.sex}' column "
                f"is equal to '{self.population.sex_m}' or '{self.population.sex_f}'. "
                f"This is not allowed in the pre-split data. \n"
                f"Invalid rows:\n{invalid_sex_rows.to_string(index=False)}"
            )

        return data

    def parse_pattern(
        self, data: DataFrame, pattern: DataFrame, model: str
    ) -> DataFrame:
        name = "While parsing pattern"

        try:
            if not all(
                col in pattern.columns
                for col in [self.pattern.val, self.pattern.val_sd]
            ):
                if not self.pattern.draws:
                    raise ValueError(
                        f"{name}: Must provide draws for pattern if pattern.val and "
                        "pattern.val_sd are not available."
                    )
                validate_columns(pattern, self.pattern.draws, name)
                pattern[self.pattern.val] = pattern[self.pattern.draws].mean(
                    axis=1
                )
                pattern[self.pattern.val_sd] = pattern[self.pattern.draws].std(
                    axis=1
                )

            validate_columns(pattern, self.pattern.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the pattern. Details:\n{e}"
            )

        pattern = pattern[self.pattern.columns].copy()

        try:
            validate_index(pattern, self.pattern.index, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: Duplicated index found in the pattern. Details:\n{e}"
            )

        try:
            validate_nonan(pattern, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: NaN values found in the pattern. Details:\n{e}"
            )

        if model == "rate":
            try:
                validate_positive(
                    pattern, [self.pattern.val, self.pattern.val_sd], name
                )
            except ValueError as e:
                raise ValueError(
                    f"{name}: Non-positive values found in 'val' or 'val_sd'. Details:\n{e}"
                )
        elif model == "logodds":
            try:
                validate_realnumber(pattern, [self.pattern.val_sd], name)
            except ValueError as e:
                raise ValueError(
                    f"{name}: Invalid real number values found. Details:\n{e}"
                )

        data_with_pattern = self._merge_with_pattern(data, pattern)
        return data_with_pattern

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "While parsing population"

        try:
            validate_columns(population, self.population.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the population data. Details:\n{e}"
            )

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

        try:
            validate_columns(data_with_population, ["m_pop", "f_pop"], name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing population columns after merging. Details:\n{e}"
            )

        try:
            validate_nonan(data_with_population, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: NaN values found in the population data. Details:\n{e}"
            )

        try:
            validate_noindexdiff(
                data, data_with_population, self.data.index, name
            )
        except ValueError as e:
            raise ValueError(
                f"{name}: Index differences found between data and population. Details:\n{e}"
            )

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
        model: Literal["rate", "logodds"] = "rate",
        output_type: Literal["rate", "count"] = "rate",
    ) -> DataFrame:
        data = self.parse_data(data)
        data = self.parse_pattern(data, pattern, model)
        data = self.parse_population(data, population)

        if output_type == "count":
            pop_normalize = False
        elif output_type == "rate":
            pop_normalize = True

        def sex_split_row(row):
            if model == "rate":
                input_pattern = np.array([1.0, row[self.pattern.val]])
                splitting_model = RateMultiplicativeModel()
            elif model == "logodds":
                # Expit of 0 is 0.5
                input_pattern = np.array([0.5, expit(row[self.pattern.val])])
                splitting_model = LogOdds_model()
            split_result, SE = split_datapoint(
                # This comes from the data
                observed_total=row[self.data.val],
                bucket_populations=np.array([row["m_pop"], row["f_pop"]]),
                # This is from sex_pattern
                rate_pattern=input_pattern,
                model=splitting_model,
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
                    "sex_split": 1,  # Indicate the row was split by sex
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

        split_df_male["sex_split"] = split_results["sex_split"]
        split_df_female["sex_split"] = split_results["sex_split"]

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
