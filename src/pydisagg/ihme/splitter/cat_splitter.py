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
    Configuration schema for categorical data DataFrame.

    This class defines the configuration parameters required to process
    a categorical dataset represented as a pandas DataFrame. It specifies
    which columns to use as indices, the categorical group for splitting,
    and the observed values along with their standard deviations.

    Parameters
    ----------
    index : List[str]
        A list of column names to be used as the index in the data DataFrame.
        These columns uniquely identify each observation in the dataset.
    cat_group : str
        The name of the column that represents the categorical group used for
        splitting the data. This column typically contains categorical or
        grouping information.
    val : str
        The name of the column that contains the observed value for each
        observation. This could be a measurement or a metric of interest.
    val_sd : str
        The name of the column that contains the standard deviation of the
        observed value, representing the uncertainty or variability
        associated with the `val` column.

    Attributes
    ----------
    index : List[str]
        As described in Parameters.
    cat_group : str
        As described in Parameters.
    val : str
        As described in Parameters.
    val_sd : str
        As described in Parameters.

    Properties
    ----------
    columns : List[str]
        A list of all required column names in the data DataFrame,
        including index columns, categorical group, value, and standard deviation columns.

    val_fields : List[str]
        A list containing the value fields, specifically `val` and `val_sd`.

    Examples
    --------
    Creating a configuration for a sample DataFrame:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from your_module import CatDataConfig  # Replace with actual module name

    >>> # Sample DataFrame
    >>> pre_split = pd.DataFrame(
    ...     {
    ...         "study_id": np.random.randint(1000, 9999, size=3),
    ...         "year_id": [2010, 2010, 2010],
    ...         "location_id": [
    ...             [1234, 1235, 1236],  # List of location_ids for row 1
    ...             [2345, 2346, 2347],  # List of location_ids for row 2
    ...             [3456],               # Single location_id for row 3 (no need to split)
    ...         ],
    ...         "mean": [0.2, 0.3, 0.4],
    ...         "std_err": [0.01, 0.02, 0.03],
    ...     }
    ... )

    >>> # Configuration
    >>> data_config = CatDataConfig(
    ...     index=["study_id", "year_id"],  # Columns to be used as index
    ...     cat_group="location_id",        # Categorical group column for splitting
    ...     val="mean",                      # Observed value column
    ...     val_sd="std_err",                # Standard deviation of the observed value
    ... )

    >>> # Accessing required columns
    >>> required_columns = data_config.columns
    >>> print(required_columns)
    ['study_id', 'year_id', 'location_id', 'mean', 'std_err']

    >>> # Accessing value fields
    >>> value_fields = data_config.val_fields
    >>> print(value_fields)
    ['mean', 'std_err']
    """

    index: List[str]
    cat_group: str
    val: str
    val_sd: str


class CatPatternConfig(Schema):
    """
    Configuration schema for the pattern DataFrame.

    This class defines the configuration parameters required to process
    a categorical pattern dataset represented as a pandas DataFrame. It specifies
    which columns to use as indices, the categorical group for splitting,
    observed mean values, their standard deviations, and additional draw columns
    if applicable.

    Parameters
    ----------
    index : List[str]
        A list of column names to be used as the index in the pattern DataFrame.
        These columns uniquely identify each pattern entry in the dataset.
    cat : str
        The name of the column that represents the categorical group used for
        splitting the data. This column typically contains categorical or
        grouping information.
    draws : List[str], optional
        A list of column names representing draw data, used for uncertainty
        quantification or simulation purposes. Defaults to an empty list.
    val : str, optional
        The name of the column that contains the observed mean value for each
        pattern entry. This could be a measurement or a metric of interest.
        Defaults to `'mean'`.
    val_sd : str, optional
        The name of the column that contains the standard deviation of the
        observed mean value, representing the uncertainty or variability
        associated with the `val` column. Defaults to `'std_err'`.
    prefix : str, optional
        A prefix to apply to column names when merging this pattern DataFrame
        with other DataFrames. This helps in distinguishing columns from different
        sources. Defaults to `'cat_pat_'`.

    Attributes
    ----------
    index : List[str]
        As described in Parameters.
    cat : str
        As described in Parameters.
    draws : List[str]
        As described in Parameters.
    val : str
        As described in Parameters.
    val_sd : str
        As described in Parameters.
    prefix : str
        As described in Parameters.

    Properties
    ----------
    columns : List[str]
        A list of all required column names in the pattern DataFrame,
        including index columns, categorical group, value, and standard deviation columns.

    val_fields : List[str]
        A list containing the value fields, specifically the `val` and `val_sd` columns.

    Examples
    --------
    Creating a configuration for a sample Pattern DataFrame:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from your_module import CatPatternConfig  # Replace with actual module name

    >>> # Sample Pattern DataFrame
    >>> all_location_ids = [
    ...     1234, 1235, 1236, 2345, 2346,
    ...     2347, 3456, 4567, 5678
    ... ]
    >>> data_pattern = pd.DataFrame(
    ...     {
    ...         "location_id": all_location_ids,
    ...         "year_id": [2010] * len(all_location_ids),
    ...         "mean": np.random.uniform(0.1, 0.5, len(all_location_ids)),
    ...         "std_err": np.random.uniform(0.01, 0.05, len(all_location_ids)),
    ...     }
    ... )

    >>> # Configuration
    >>> pattern_config = CatPatternConfig(
    ...     index=["year_id"],               # Columns to be used as index
    ...     cat="location_id",               # Categorical group column for splitting
    ...     draws=[],                         # No draw columns in this example
    ...     val="mean",                      # Observed mean value column
    ...     val_sd="std_err",                # Standard deviation of the observed mean value
    ...     prefix="cat_pat_"                 # Prefix for merging
    ... )

    >>> # Accessing required columns
    >>> required_columns = pattern_config.columns
    >>> print(required_columns)
    ['year_id', 'location_id', 'mean', 'std_err']

    >>> # Accessing value fields
    >>> value_fields = pattern_config.val_fields
    >>> print(value_fields)
    ['mean', 'std_err']
    """

    index: List[str]
    cat: str
    draws: List[str] = []
    val: str = "mean"
    val_sd: str = "std_err"
    prefix: str = "cat_pat_"


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
    # target: str
    val: str
    prefix: str = "cat_pop_"


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
                "Meow! The pattern's match criteria must be a subset of the data. Purrrlease check your input."
            )
        if not set(self.population.index).issubset(
            self.data.index + self.pattern.index
        ):
            raise ValueError(
                "Meow! The population's match criteria must be a subset of the data and the pattern. Purrrhaps take a closer look?"
            )
        # NOTE: This doesn't have to be true, as long as the values contained within the group are present in the pattern
        if self.pattern.cat != self.data.cat_group:
            raise ValueError(
                "Hiss! The 'target' column in the pattern doesn't match the 'target' column in the data. Meow over it again."
            )
        if self.data.cat_group not in self.population.index:
            raise ValueError(
                "Meow! The 'target' column in the population must match the 'target' column in the data. Purr-fect that before proceeding!"
            )

    # def create_ref_return_df(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
    #     """
    #     Create reference and return DataFrames.

    #     Parameters
    #     ----------
    #     data : DataFrame
    #         The input data DataFrame.

    #     Returns
    #     -------
    #     tuple[DataFrame, DataFrame]
    #         A tuple containing:
    #         - ref_df: DataFrame with original data, exploded if necessary.
    #         - data: DataFrame with required columns and identifiers.
    #     """
    #     ref_df = data.copy()
    #     ref_df["orig_pyd_id"] = range(
    #         len(ref_df)
    #     )  # Assign original pyd_id before exploding
    #     ref_df["orig_group"] = ref_df[self.data.cat_group]
    #     # Explode the 'target' column if it contains lists
    #     if ref_df[self.data.cat_group].apply(lambda x: isinstance(x, list)).any():
    #         ref_df = ref_df.explode(self.data.cat_group).reset_index(drop=True)
    #     # Assign new pyd_id's after exploding
    #     ref_df["pyd_id"] = range(len(ref_df))
    #     return ref_df, ref_df[self.data.columns + ["pyd_id", "orig_pyd_id"]]

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
        validate_columns(data, self.data.columns, name)

        validate_index(data, self.data.index + [self.data.cat_group], name)

        validate_nonan(data, name)

        validate_positive(data, [self.data.val, self.data.val_sd], name)

        return data

    def _merge_with_pattern(
        self,
        data: DataFrame,
        pattern: DataFrame,
        how: Literal["left", "right", "outer", "inner"],
    ) -> DataFrame:
        """
        Merge data with pattern DataFrame.

        Parameters
        ----------
        data : DataFrame
            The data DataFrame.
        pattern : DataFrame
            The pattern DataFrame.
        how : {'inner', 'left', 'right', 'outer'}
            Merge method.

        Returns
        -------
        DataFrame
            Merged DataFrame after merging with pattern.
        """
        merge_keys = self.pattern.index + [self.pattern.cat]
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
        data_target_ids = data[self.data.cat_group].unique()
        pattern_copy = pattern_copy[
            pattern_copy[self.pattern.cat].isin(data_target_ids)
        ]

        # Use an inner join
        data_with_pattern = self._merge_with_pattern(data, pattern_copy, how="inner")

        # Validate index differences after merging
        validate_noindexdiff(
            data,
            data_with_pattern,
            self.data.index + [self.data.cat_group],
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

        # NOTE: Updated to this point

        # Should we error sooner if a population is missing for the data?
        # How should we check instead of looping?

        # This isn't right I don't think ...
        data_target_ids = data[self.data.cat_group].unique()

        population = population[
            population[self.population.target].isin(data_target_ids)
        ]

        # Use an inner join
        merge_keys = self.population.index
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
            self.data.index + [self.data.cat_group, "pyd_id"],
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
        use_parallel: bool = False,  # Option to run in parallel
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
