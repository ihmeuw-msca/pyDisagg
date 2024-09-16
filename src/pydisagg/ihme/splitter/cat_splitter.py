from typing import Any, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from scipy.special import expit  # type: ignore
from typing import Literal
from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.schema import Schema  # Assuming your original Schema class
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
    index: List[str]
    target: str
    sub_target: str  # Column that contains list of sub-targets
    val: str
    val_sd: str

    @property
    def columns(self) -> List[str]:
        return list(
            set(
                self.index
                + [self.target, self.sub_target, self.val, self.val_sd]
            )
        )

    @property
    def val_fields(self) -> List[str]:
        return ["val", "val_sd"]  # Attribute names


class CatPatternConfig(Schema):
    index: List[str]
    sub_target: str
    draws: List[str] = []
    val: str = "mean"
    val_sd: str = "std_err"
    prefix: str = "cat_pat_"

    @property
    def columns(self) -> List[str]:
        return list(set(self.index + [self.sub_target, self.val, self.val_sd]))

    @property
    def val_fields(self) -> List[str]:
        return ["val", "val_sd"]  # Attribute names


class CatPopulationConfig(Schema):
    index: List[str]
    sub_target: str
    val: str
    prefix: str = "cat_pop_"

    @property
    def columns(self) -> List[str]:
        return list(set(self.index + [self.sub_target, self.val]))

    @property
    def val_fields(self) -> List[str]:
        return ["val"]


class CatSplitter(BaseModel):
    data: CatDataConfig
    pattern: CatPatternConfig
    population: CatPopulationConfig

    def model_post_init(self, __context: Any) -> None:
        """Extra validation of all the matches."""
        if not set(self.pattern.index).issubset(self.data.index):
            raise ValueError(
                "Match criteria in the pattern must be a subset of the data"
            )
        if not set(self.population.index).issubset(
            self.data.index + self.pattern.index
        ):
            raise ValueError(
                "Match criteria in the population must be a subset of the data and the pattern"
            )

    def create_ref_return_df(
        self, data: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
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
            raise KeyError(
                f"{name}: Missing columns in the input data. Details:\n{e}"
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

        # Explode the 'sub_target' column if it contains lists
        if (
            data[self.data.sub_target]
            .apply(lambda x: isinstance(x, list))
            .any()
        ):
            data = data.explode(self.data.sub_target).reset_index(drop=True)
            # Rename the sub_target column to match the pattern's sub_target if necessary
            if self.data.sub_target != self.pattern.sub_target:
                data.rename(
                    columns={self.data.sub_target: self.pattern.sub_target},
                    inplace=True,
                )
                self.data.sub_target = self.pattern.sub_target

        return data

    def _merge_with_pattern(
        self, data: DataFrame, pattern: DataFrame
    ) -> DataFrame:
        merge_keys = self.pattern.index + [self.pattern.sub_target]
        val_fields = [
            getattr(self.pattern, field) for field in self.pattern.val_fields
        ]
        data_with_pattern = data.merge(
            pattern, on=merge_keys, how="left"
        ).dropna(subset=val_fields)
        return data_with_pattern

    def parse_pattern(
        self, data: DataFrame, pattern: DataFrame, model: str
    ) -> DataFrame:
        name = "While parsing pattern"

        try:
            val_cols = [self.pattern.val, self.pattern.val_sd]
            if not all(col in pattern.columns for col in val_cols):
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

        pattern_copy = pattern.copy()
        pattern_copy = pattern_copy[self.pattern.columns]
        rename_map = self.pattern.apply_prefix()
        pattern_copy.rename(columns=rename_map, inplace=True)

        data_with_pattern = self._merge_with_pattern(data, pattern_copy)

        # Validate index differences after merging
        validate_noindexdiff(data, data_with_pattern, self.data.index, name)

        return data_with_pattern

    def parse_population(
        self, data: DataFrame, population: DataFrame
    ) -> DataFrame:
        name = "While parsing population"

        # Validate population columns
        try:
            validate_columns(population, self.population.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the population data. Details:\n{e}"
            )

        # Rename sub_target in population to match data if necessary
        if self.population.sub_target != self.data.sub_target:
            population = population.rename(
                columns={self.population.sub_target: self.data.sub_target}
            )
            self.population.sub_target = self.data.sub_target

        # Merge population data with main data
        merge_keys = self.population.index + [self.population.sub_target]
        val_fields = [
            getattr(self.population, field)
            for field in self.population.val_fields
        ]
        data_with_population = data.merge(
            population,
            on=merge_keys,
            how="left",
            suffixes=("", "_pop"),
        )

        # Validate for NaN values
        try:
            validate_nonan(data_with_population, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: NaN values found in the population data. Details:\n{e}"
            )

        # Validate index differences
        validate_noindexdiff(data, data_with_population, self.data.index, name)

        # Ensure the population column is numeric
        data_with_population[self.population.val] = data_with_population[
            self.population.val
        ].astype("float64")

        return data_with_population

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: Literal["rate", "logodds"] = "rate",
        output_type: Literal["rate", "count"] = "rate",
    ) -> DataFrame:
        """
        Split the input data based on a specified pattern and population model.
        """

        # Parsing input data, pattern, and population
        ref_df, data = self.create_ref_return_df(data)
        data = self.parse_data(data)
        data = self.parse_pattern(data, pattern, model)
        data = self.parse_population(data, population)

        # Determine whether to normalize by population for the output type
        pop_normalize = output_type == "rate"

        # Handle rows where 'sub_target' == 'target' (no need to split)
        mask_no_split = data[self.data.sub_target] == data[self.data.target]

        # Create a copy for the final DataFrame where rows are not split
        final_df = data[mask_no_split].copy()

        # Set the result columns for non-split rows
        final_df["split_result"] = final_df[self.data.val]
        final_df["split_result_se"] = final_df[self.data.val_sd]
        final_df["split_flag"] = 0  # Mark as not split

        # Handle rows that need to be split
        split_data = data[~mask_no_split].copy()

        # Group by the original rows using 'pyd_id'
        split_results = []
        for pyd_id, group in split_data.groupby("pyd_id"):
            observed_total = group[self.data.val].iloc[0]
            observed_total_se = group[self.data.val_sd].iloc[0]
            bucket_populations = group[self.population.val].values
            rate_pattern = group[self.pattern.val].values
            pattern_sd = group[self.pattern.val_sd].values
            pattern_covariance = np.diag(pattern_sd**2)

            if model == "rate":
                splitting_model = RateMultiplicativeModel()
            elif model == "logodds":
                splitting_model = LogOddsModel()

            # Perform splitting
            split_result, split_se = split_datapoint(
                observed_total=observed_total,
                bucket_populations=bucket_populations,
                rate_pattern=rate_pattern,
                model=splitting_model,
                output_type=output_type,
                normalize_pop_for_average_type_obs=pop_normalize,
                observed_total_se=observed_total_se,
                pattern_covariance=pattern_covariance,
            )

            # Assign results back to the group
            group["split_result"] = split_result
            group["split_result_se"] = split_se
            group["split_flag"] = 1
            split_results.append(group)

        # Concatenate the split results
        if split_results:
            split_df = pd.concat(split_results, ignore_index=True)
            # Combine the non-split rows and the split rows
            final_split_df = pd.concat([final_df, split_df], ignore_index=True)
        else:
            final_split_df = final_df.copy()

        # Merge back with ref_df to restore original columns
        final_split_df = final_split_df.merge(
            ref_df, on="pyd_id", how="left", suffixes=("", "_orig")
        )

        # Drop the '_orig' columns if they were added
        final_split_df = final_split_df.loc[
            :, ~final_split_df.columns.str.endswith("_orig")
        ]

        # Remove temporary columns
        final_split_df.drop(columns=["pyd_id"], inplace=True)

        return final_split_df
