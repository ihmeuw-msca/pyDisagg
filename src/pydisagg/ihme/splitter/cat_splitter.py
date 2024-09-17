from typing import Any, List

import numpy as np
import pandas as pd
import multiprocessing
from pandas import DataFrame
from pydantic import BaseModel
from typing import Literal
from joblib import Parallel, delayed

from pydisagg.disaggregate import split_datapoint
from pydisagg.ihme.schema import Schema
from pydisagg.models import RateMultiplicativeModel, LogOddsModel
from pydisagg.ihme.validator import (
    validate_columns,
    validate_index,
    validate_noindexdiff,
    validate_nonan,
    validate_positive,
)


class CatDataConfig(Schema):
    """
    Configuration for the data DataFrame.

    Parameters
    ----------
    index : List[str]
        List of column names to be used as index in the data DataFrame.
    target : str
        Column name representing the target variable to split.
    val : str
        Column name for the observed value.
    val_sd : str
        Column name for the standard deviation of the observed value.
    """

    index: List[str]
    target: str
    val: str
    val_sd: str

    @property
    def columns(self) -> List[str]:
        """
        List of all required columns in the data DataFrame.

        Returns
        -------
        List[str]
            List of column names.
        """
        return list(set(self.index + [self.target, self.val, self.val_sd]))

    @property
    def val_fields(self) -> List[str]:
        """
        List of value fields (attributes).

        Returns
        -------
        List[str]
            List containing 'val' and 'val_sd'.
        """
        return ["val", "val_sd"]  # Attribute names


class CatPatternConfig(Schema):
    """
    Configuration for the pattern DataFrame.

    Parameters
    ----------
    index : List[str]
        List of column names to be used as index in the pattern DataFrame.
    target : str
        Column name representing the target variable to split.
    draws : List[str], optional
        List of draw column names, by default [].
    val : str, optional
        Column name for the mean value in the pattern DataFrame, by default 'mean'.
    val_sd : str, optional
        Column name for the standard deviation in the pattern DataFrame, by default 'std_err'.
    prefix : str, optional
        Prefix to apply to column names when merging, by default 'cat_pat_'.
    """

    index: List[str]
    target: str
    draws: List[str] = []
    val: str = "mean"
    val_sd: str = "std_err"
    prefix: str = "cat_pat_"

    @property
    def columns(self) -> List[str]:
        """
        List of all required columns in the pattern DataFrame.

        Returns
        -------
        List[str]
            List of column names.
        """
        return list(set(self.index + [self.target, self.val, self.val_sd]))

    @property
    def val_fields(self) -> List[str]:
        """
        List of value fields (attributes).

        Returns
        -------
        List[str]
            List containing 'val' and 'val_sd'.
        """
        return ["val", "val_sd"]  # Attribute names

    def apply_prefix(self) -> dict:
        """
        Create a mapping to rename columns with the specified prefix.

        Returns
        -------
        dict
            Mapping from original column names to prefixed column names.
        """
        return {
            self.val: f"{self.prefix}{self.val}",
            self.val_sd: f"{self.prefix}{self.val_sd}",
            self.target: self.target,  # Do not prefix the target column
            **{idx: idx for idx in self.index},  # Keep index columns unchanged
        }


class CatPopulationConfig(Schema):
    """
    Configuration for the population DataFrame.

    Parameters
    ----------
    index : List[str]
        List of column names to be used as index in the population DataFrame.
    target : str
        Column name representing the target variable to split.
    val : str
        Column name for the population value.
    prefix : str, optional
        Prefix to apply to column names when merging, by default 'cat_pop_'.
    """

    index: List[str]
    target: str
    val: str
    prefix: str = "cat_pop_"

    @property
    def columns(self) -> List[str]:
        """
        List of all required columns in the population DataFrame.

        Returns
        -------
        List[str]
            List of column names.
        """
        return list(set(self.index + [self.target, self.val]))

    @property
    def val_fields(self) -> List[str]:
        """
        List of value fields (attributes).

        Returns
        -------
        List[str]
            List containing 'val'.
        """
        return ["val"]

    def apply_prefix(self) -> dict:
        """
        Create a mapping to rename columns with the specified prefix.

        Returns
        -------
        dict
            Mapping from original column names to prefixed column names.
        """
        return {
            self.val: f"{self.prefix}{self.val}",
            self.target: self.target,  # Do not prefix the target column
            **{idx: idx for idx in self.index},  # Keep index columns unchanged
        }


class CatSplitter(BaseModel):
    """
    Class for splitting categorical data based on pattern and population data.

    Parameters
    ----------
    data : CatDataConfig
        Configuration for the data DataFrame.
    pattern : CatPatternConfig
        Configuration for the pattern DataFrame.
    population : CatPopulationConfig
        Configuration for the population DataFrame.
    """

    data: CatDataConfig
    pattern: CatPatternConfig
    population: CatPopulationConfig

    def model_post_init(self, __context: Any) -> None:
        """
        Perform extra validation after model initialization.

        Raises
        ------
        ValueError
            If the match criteria in the pattern or population do not match the data.
        """
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
        # Check that the 'target' column in pattern and population matches data
        if self.pattern.target != self.data.target:
            raise ValueError(
                "The 'target' column in pattern must match the 'target' column in data"
            )
        if self.population.target != self.data.target:
            raise ValueError(
                "The 'target' column in population must match the 'target' column in data"
            )

    def create_ref_return_df(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
        """
        Create reference and return DataFrames.

        Parameters
        ----------
        data : DataFrame
            The input data DataFrame.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing:
            - ref_df: DataFrame with original data, exploded if necessary.
            - data: DataFrame with required columns and identifiers.
        """
        ref_df = data.copy()
        ref_df["orig_pyd_id"] = range(
            len(ref_df)
        )  # Assign original pyd_id before exploding
        ref_df["orig_group"] = ref_df[self.data.target]
        # Explode the 'target' column if it contains lists
        if ref_df[self.data.target].apply(lambda x: isinstance(x, list)).any():
            ref_df = ref_df.explode(self.data.target).reset_index(drop=True)
        # Assign new pyd_id's after exploding
        ref_df["pyd_id"] = range(len(ref_df))
        return ref_df, ref_df[self.data.columns + ["pyd_id", "orig_pyd_id"]]

    def parse_data(self, data: DataFrame) -> DataFrame:
        """
        Parse and validate the input data DataFrame.

        Parameters
        ----------
        data : DataFrame
            The input data DataFrame.

        Returns
        -------
        DataFrame
            Validated and possibly modified data DataFrame.

        Raises
        ------
        KeyError
            If required columns are missing.
        ValueError
            If there are duplicate indices, NaN values, or non-positive values.
        """
        name = "While parsing data"

        # Validate core columns first
        try:
            validate_columns(data, self.data.columns + ["pyd_id", "orig_pyd_id"], name)
        except KeyError as e:
            raise KeyError(f"{name}: Missing columns in the input data. Details:\n{e}")

        try:
            validate_index(data, self.data.index + [self.data.target, "pyd_id"], name)
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

        return data

    def _merge_with_pattern(
        self, data: DataFrame, pattern: DataFrame, how: str
    ) -> DataFrame:
        """
        Merge data with pattern DataFrame.

        Parameters
        ----------
        data : DataFrame
            The data DataFrame.
        pattern : DataFrame
            The pattern DataFrame.
        how : str
            Merge method ('inner', 'left', 'right', etc.).

        Returns
        -------
        DataFrame
            Merged DataFrame after merging with pattern.
        """
        merge_keys = self.pattern.index + [self.pattern.target]
        val_fields = [
            self.pattern.apply_prefix()[self.pattern.val],
            self.pattern.apply_prefix()[self.pattern.val_sd],
        ]
        data_with_pattern = data.merge(pattern, on=merge_keys, how=how).dropna(
            subset=val_fields
        )
        return data_with_pattern

    def parse_pattern(
        self, data: DataFrame, pattern: DataFrame, model: str
    ) -> DataFrame:
        """
        Parse and merge the pattern DataFrame with data.

        Parameters
        ----------
        data : DataFrame
            The data DataFrame.
        pattern : DataFrame
            The pattern DataFrame.
        model : str
            The model type ('rate' or 'logodds').

        Returns
        -------
        DataFrame
            DataFrame after merging with pattern data.

        Raises
        ------
        KeyError
            If required columns are missing in pattern.
        ValueError
            If necessary columns or draws are not provided.
        """
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
                pattern[self.pattern.val] = pattern[self.pattern.draws].mean(axis=1)
                pattern[self.pattern.val_sd] = pattern[self.pattern.draws].std(axis=1)

            validate_columns(pattern, self.pattern.columns, name)
        except KeyError as e:
            raise KeyError(f"{name}: Missing columns in the pattern. Details:\n{e}")

        pattern_copy = pattern.copy()
        pattern_copy = pattern_copy[self.pattern.columns]
        rename_map = self.pattern.apply_prefix()
        pattern_copy.rename(columns=rename_map, inplace=True)

        # Filter pattern_copy to include only target IDs present in data
        data_target_ids = data[self.data.target].unique()
        pattern_copy = pattern_copy[
            pattern_copy[self.pattern.target].isin(data_target_ids)
        ]

        # Use an inner join
        data_with_pattern = self._merge_with_pattern(data, pattern_copy, how="inner")

        # Validate index differences after merging
        validate_noindexdiff(
            data,
            data_with_pattern,
            self.data.index + [self.data.target, "pyd_id"],
            name,
        )

        return data_with_pattern

    def parse_population(self, data: DataFrame, population: DataFrame) -> DataFrame:
        """
        Parse and merge the population DataFrame with data.

        Parameters
        ----------
        data : DataFrame
            The data DataFrame.
        population : DataFrame
            The population DataFrame.

        Returns
        -------
        DataFrame
            DataFrame after merging with population data.

        Raises
        ------
        KeyError
            If required columns are missing in population.
        ValueError
            If NaN values are found after merging.
        """
        name = "While parsing population"

        # Validate population columns
        try:
            validate_columns(population, self.population.columns, name)
        except KeyError as e:
            raise KeyError(
                f"{name}: Missing columns in the population data. Details:\n{e}"
            )

        # Filter population to include only target IDs present in data
        data_target_ids = data[self.data.target].unique()
        population = population[
            population[self.population.target].isin(data_target_ids)
        ]

        # Use an inner join
        merge_keys = self.population.index + [self.population.target]
        data_with_population = data.merge(
            population, on=merge_keys, how="inner", suffixes=("", "_pop")
        )

        # Validate for NaN values
        try:
            validate_nonan(data_with_population, name)
        except ValueError as e:
            raise ValueError(
                f"{name}: NaN values found in the population data. Details:\n{e}"
            )

        # Validate index differences
        validate_noindexdiff(
            data,
            data_with_population,
            self.data.index + [self.data.target, "pyd_id"],
            name,
        )

        # Ensure the population column is numeric
        data_with_population[self.population.val] = data_with_population[
            self.population.val
        ].astype("float64")

        return data_with_population

    def _process_group(
        self, group: DataFrame, model: str, output_type: str
    ) -> DataFrame:
        """
        Process a group of data for splitting.

        Parameters
        ----------
        group : DataFrame
            The group of data to process.
        model : str
            The model type ('rate' or 'logodds').
        output_type : str
            The output type ('rate' or 'count').

        Returns
        -------
        DataFrame
            The processed group with splitting results.
        """
        observed_total = group[self.data.val].iloc[0]
        observed_total_se = group[self.data.val_sd].iloc[0]

        if len(group) == 1:
            # No need to split, assign the observed values
            group["split_result"] = observed_total
            group["split_result_se"] = observed_total_se
            group["split_flag"] = 0  # Not split
        else:
            # Need to split among multiple targets
            bucket_populations = group[self.population.val].values
            rate_pattern = group[self.pattern.apply_prefix()[self.pattern.val]].values
            pattern_sd = group[self.pattern.apply_prefix()[self.pattern.val_sd]].values
            pattern_covariance = np.diag(pattern_sd**2)

            if model == "rate":
                splitting_model = RateMultiplicativeModel()
            elif model == "logodds":
                splitting_model = LogOddsModel()

            # Determine whether to normalize by population for the output type
            pop_normalize = output_type == "rate"

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
            group["split_flag"] = 1  # Split

        return group

    def split(
        self,
        data: DataFrame,
        pattern: DataFrame,
        population: DataFrame,
        model: Literal["rate", "logodds"] = "rate",
        output_type: Literal["rate", "count"] = "rate",
        n_jobs: int = -1,  # Use all available cores by default
        use_parallel: bool = True,  # Option to run in parallel
    ) -> DataFrame:
        """
        Split the input data based on a specified pattern and population model.

        Parameters
        ----------
        data : DataFrame
            The input data DataFrame.
        pattern : DataFrame
            The pattern DataFrame.
        population : DataFrame
            The population DataFrame.
        model : {'rate', 'logodds'}, optional
            The model to use for splitting, by default 'rate'.
        output_type : {'rate', 'count'}, optional
            The output type desired, by default 'rate'.
        n_jobs : int, optional
            Number of jobs for parallel processing, by default -1 (use all available cores).
        use_parallel : bool, optional
            Whether to use parallel processing, by default True.

        Returns
        -------
        DataFrame
            DataFrame containing the split results.

        Raises
        ------
        ValueError
            If validation fails during parsing.
        """
        # Parsing input data, pattern, and population
        ref_df, data = self.create_ref_return_df(data)

        # Keep track of columns not used in the analysis
        all_columns = ref_df.columns.tolist()
        columns_used = self.data.columns + ["pyd_id", "orig_pyd_id"]
        columns_not_used = list(set(all_columns) - set(columns_used))

        data = self.parse_data(data)
        data = self.parse_pattern(data, pattern, model)
        data = self.parse_population(data, population)

        # Process groups
        if use_parallel:
            # Identify unique 'orig_pyd_id's to process
            num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            processed_groups = Parallel(n_jobs=num_cores, backend="loky")(
                delayed(self._process_group)(group, model, output_type)
                for _, group in data.groupby("orig_pyd_id")
            )

            # Concatenate the results
            final_split_df = pd.concat(processed_groups, ignore_index=True)
        else:
            # Process groups using regular groupby
            final_split_df = (
                data.groupby("orig_pyd_id", group_keys=False)
                .apply(lambda group: self._process_group(group, model, output_type))
                .reset_index(drop=True)
            )

        # Merge back only columns not used in the analysis
        if columns_not_used:
            final_split_df = final_split_df.merge(
                ref_df[["pyd_id"] + columns_not_used],
                on="pyd_id",
                how="left",
            )

        # Remove temporary columns
        final_split_df.drop(columns=["pyd_id", "orig_pyd_id"], inplace=True)

        return final_split_df
