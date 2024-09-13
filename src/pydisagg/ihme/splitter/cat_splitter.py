# Motivating example:
# Trying to split state data into county data
# - Data:
#   - Match - the criteria beyond the reference column that will be used to establish which data to match the split (i.e. year_id, sex_id, age_group_id)
#   - target - the thing that wants to be split (not the value), e.g. state (Iowa)
#   - val - the value that will be split (e.g. population)
#   - val_sd - the standard deviation of the value that will be split

# - Pattern:
#   - Match - same as above
#   - target - the thing that wants to be split (not the value), e.g. state (Iowa)
#   - sub_target - the thing that will inform the split (e.g. county)
#   - draws - can be used instead of val and val_sd
#   - val - the value that will be split (e.g. population)
#       - Would this be in the same space as the pre-split value?
#       - Iowa estimates 0.3 prevalence of a disease, what would need to be given for the counties?
#   - val_sd - the standard deviation of the value that will be split

# - Population:
#   - Match - same as above
#   - target - the thing that wants to be split (not the value), e.g. state (Iowa)
#   - sub_target - the thing that will inform the split and link to target
#   - val - population for th


from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from scipy.special import expit  # type: ignore
from typing import Literal
from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.schema import Schema
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_noindexdiff,
    validate_nonan,
    validate_positive,
    validate_realnumber,
)
from pydisagg.models import RateMultiplicativeModel
from pydisagg.models import LogOddsModel


class CatDataConfig(Schema):
    match: list[str]
    target: str
    sub_target: str  # We are going to assume that the sub_target column will have a list of values per row that associate with the target. If target == sub_target that is the reference for that group?
    val: str
    val_sd: str

    @property
    def columns(self) -> list[str]:
        return list(set(self.match + [self.target, self.val, self.val_sd]))


class CatPatternConfig(Schema):
    match: list[str]
    target: str
    sub_target: str
    draws: list[str] = []
    val: str
    val_sd: str
    prefix: str = "cat_pat_"

    @property
    def columns(self) -> list[str]:
        return list(
            set(self.match + [self.target, self.sub_target, self.val, self.val_sd])
        )


class CatPopulationConfig(Schema):
    match: list[str]
    target: str
    sub_target: str
    val: str
    prefix: str = "cat_pop_"


class CatSplitter(BaseModel):
    data: CatDataConfig
    pattern: CatPatternConfig
    population: CatPopulationConfig

    def model_post_init(self, __context: Any) -> None:
        """Extra validation of all the matches."""
        if not set(self.pattern.match).issubset(self.data.match):
            raise ValueError(
                "Match criteria in the pattern must be a subset of the data"
            )
        if not set(self.population.match).issubset(
            self.data.match + self.pattern.match
        ):
            raise ValueError(
                "Match criteria in the population must be a subset of the data and the pattern"
            )

    def create_ref_return_df(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
        """
        Create and return two DataFrames: one with all original columns, and another with only relevant columns.

        Parameters
        ----------
        data : DataFrame
            The input DataFrame containing the original data from which the two DataFrames will be created.

        Returns
        -------
        ref_df : DataFrame
            A DataFrame that contains all the original columns from `data`, along with an additional `pyd_id` column
            that assigns a unique identifier to each row (ranging from 0 to nrows-1).

        return_df : DataFrame
            A DataFrame that contains only the relevant columns (as defined in the configuration) plus the `pyd_id` column.
            The relevant columns are determined by the `self.data.columns` or other relevant configuration.

        Notes
        -----
        - The `pyd_id` column ensures row uniqueness and helps with diagnostics.
        - `return_df` is intended to simplify analysis by limiting the DataFrame to only the columns of interest.
        - The relevant columns are retrieved from the configuration (e.g., `self.data.columns`), which may depend
        on the specific configuration being used.

        """
        # Expensive space-wise, but helps ensure row uniqueness/ additional checks
        ref_df = data.copy()
        ref_df["pyd_id"] = range(len(ref_df))
        return_df = ref_df[self.data.columns + ["pyd_id"]]

        return ref_df, return_df

    def parse_data(self, data: DataFrame) -> DataFrame:
        name = "While parsing data"

        # Validate core columns first
        try:
            validate_columns(data, self.data.columns, name)
        except KeyError as e:
            raise KeyError(f"{name}: Missing columns in the input data. Details:\n{e}")

        if self.population.target not in data.columns:
            raise KeyError(
                f"{name}: Missing column '{self.population.target}' in the input data."
            )

        try:
            validate_index(data, self.data.match, name)
        except ValueError as e:
            raise ValueError(f"{name}: Duplicated index found. Details:\n{e}")

        try:
            validate_nonan(data, name)
        except ValueError as e:
            raise ValueError(f"{name}: NaN values found. Details:\n{e}")

        try:
            # Does the value need to be positive to split?
            validate_positive(data, [self.data.val, self.data.val_sd], name)
        except ValueError as e:
            raise ValueError(
                f"{name}: Non-positive values found in 'val' or 'val_sd'. Details:\n{e}"
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
                pattern[self.pattern.val] = pattern[self.pattern.draws].mean(axis=1)
                pattern[self.pattern.val_sd] = pattern[self.pattern.draws].std(axis=1)

            validate_columns(pattern, self.pattern.columns, name)
        except KeyError as e:
            raise KeyError(f"{name}: Missing columns in the pattern. Details:\n{e}")

        try:
            validate_index(pattern, self.pattern.match, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: Duplicated index found in the pattern. Details:\n{e}"
            )

        try:
            validate_nonan(pattern, name)
        except ValueError as e:
            raise ValueError(f"{name}: NaN values found in the pattern. Details:\n{e}")

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

        pattern_copy = pattern.copy()
        pattern_copy = pattern_copy[self.pattern.columns]
        rename_map = self.pattern.apply_prefix()
        pattern_copy.rename(columns=rename_map, inplace=True)

        data_with_pattern = self._merge_with_pattern(data, pattern_copy)

        # Validate index differences after merging
        validate_noindexdiff(data, data_with_pattern, self.data.match, name)

        return data_with_pattern

    def parse_population(self, data: DataFrame, population: DataFrame) -> DataFrame:
        name = "While parsing population"

        # Step 1: Validate population columns
        try:
            validate_columns(population, self.population.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the population data. Details:\n{e}"
            )

        # Step 2: Get all the population data for a given target and match
        # we have target and sub_target in the population data, e.g. target = state, sub_target = county
        # so for each target, we want to group the sub targets and get a relative proportion of the population
        # We should probably do this for target-match combination so that we dont have to recalculate the proportions

        ### Progress so far
        #
        #
        #
        #

        male_population = self.get_population_by_sex(population, self.population.sex_m)
        female_population = self.get_population_by_sex(
            population, self.population.sex_f
        )

        male_population.rename(columns={self.population.val: "m_pop"}, inplace=True)
        female_population.rename(columns={self.population.val: "f_pop"}, inplace=True)

        # Step 3: Merge population data with main data
        data_with_population = self._merge_with_population(
            data, male_population, "m_pop"
        )
        data_with_population = self._merge_with_population(
            data_with_population, female_population, "f_pop"
        )

        # Step 4: Validate the merged data columns
        try:
            validate_columns(data_with_population, ["m_pop", "f_pop"], name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing population columns after merging. Details:\n{e}"
            )

        # Step 5: Validate for NaN values
        try:
            validate_nonan(data_with_population, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: NaN values found in the population data. Details:\n{e}"
            )

        # Step 6: Validate index differences
        try:
            validate_noindexdiff(data, data_with_population, self.data.index, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: Index differences found between data and population. Details:\n{e}"
            )

        # Ensure the columns are in the correct numeric type (e.g., float64)
        # Convert "m_pop" and "f_pop" columns to standard numeric types if necessary
        data_with_population["m_pop"] = data_with_population["m_pop"].astype("float64")
        data_with_population["f_pop"] = data_with_population["f_pop"].astype("float64")

        return data_with_population

    def _merge_with_pattern(self, data: DataFrame, pattern: DataFrame) -> DataFrame:
        data_with_pattern = data.merge(
            pattern, on=self.pattern.match, how="left"
        ).dropna()
        return data_with_pattern
