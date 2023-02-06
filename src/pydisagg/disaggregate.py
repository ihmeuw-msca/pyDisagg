"""Module containing high level api for splitting"""
from typing import Optional, Union

import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from pydisagg.models import LMO_model, LogOdds_model
from pydisagg.DisaggModel import DisaggModel


def split_datapoint(
    observed_total: float,
    bucket_populations: NDArray,
    rate_pattern: NDArray,
    observed_total_se: Optional[float] = None,
    model: Optional[DisaggModel] = LogOdds_model(),
    CI_method: Optional[str] = 'delta-wald'
) -> Union[tuple,NDArray]:
    """Disaggregate a datapoint using the model given as input.
    Defaults to assuming multiplicativity in the odds ratio

    Parameters
    ----------
    observed_total : float
        aggregated observed_total across all buckets, value to be split
    bucket_populations : NDArray
        population size in each bucket
    rate_pattern : NDArray
        Rate Pattern to use, should be an estimate of the rates in each bucket
            that we want to rescale
    observed_total_se : Optional[float], optional
        standard error of observed_total, by default None
    model : Optional[DisaggModel], optional
        DisaggModel to use, by default LMO_model(1)
    CI_method : Optional[str], optional
        method to use for confidence intervals,
        see documentation for standard error methods in DisaggModel, by default 'delta-wald'

    Returns
    -------
    Union[Tuple,NDArray]
        If standard errors are available, this will return the tuple
            (
                estimate_in_each_bucket,
                se_of_estimate_bucket,
                (CI_lower_in_each_bucket,CI_upper_in_each_bucket)
            )
        Otherwise, if standard errors are not available, 
        this will return a numpy array of the disaggregated estimates

    Notes
    -----
    If no observed_total_se is given, returns point estimates
    If observed_total_se is given, then returns a tuple
        (point_estimate,standard_error,(CI_lower,CI_upper))
    """
    return model.split_groups(
        bucket_populations,
        observed_total,
        observed_total_se,
        rate_pattern,
        CI_method=CI_method
    )


def split_dataframe(
    groups_to_split_into: list,
    observation_group_membership_df: DataFrame,
    population_sizes: DataFrame,
    rate_patterns: DataFrame,
    use_se: Optional[bool] = False,
    model: Optional[DisaggModel] = LMO_model(1),
) -> DataFrame:
    """Disaggregate datapoints and pivots observations into estimates for each group per pop id

    Parameters
    ----------
    groups_to_split_into : list
        list of groups to disaggregate observations into
    observation_group_membership_df : DataFrame
        Dataframe with columns location_id, pattern_id, obs,
        and columns for each of the groups_to_split_into
        with dummy variables that represent whether or not
        each group is included in the observations for that row.
        This also optionally contains a obs_se column which will be used if use_se is True
        location_id represents the population that the observation comes from
        pattern_id gives the baseline that should be used for splitting
    population_sizes : DataFrame
        Dataframe with location_id as the index containing the
        size of each group within each population (given the location_id)
    rate_patterns : DataFrame
        dataframe with pattern_id as the index, and columns
        for each of the groups_to_split where the entries represent the rate pattern
        in the given group to use for pydisagg.
    use_se : Optional[bool], optional
        whether or not to report standard errors along with estimates
        if set to True, then observation_group_membership_df must have an obs_se column
        , by default False
    model : Optional[DisaggModel], optional
        DisaggModel to use for splitting, by default LMO_model(1)

    Returns
    -------
    DataFrame
        Dataframe where each row corresponds to one of obs, with one or
            two columns for each of the groups_to_split_into, giving the estimate
        If use_se==True, then has a nested column indexing, where both the
            point estimate and standard error for the estimate for each group is given.
    """
    splitting_df = observation_group_membership_df.copy()
    if use_se is False:
        def split_row(x):
            return split_datapoint(
                x['obs'],
                population_sizes.loc[x.name]*x[groups_to_split_into],
                rate_patterns.loc[x['pattern_id']],
                model=model
            )
        result = (
            splitting_df
            .set_index('location_id')
            .apply(
                split_row,
                axis=1)
            .reset_index()
            # .groupby('location_id')
            # .sum()
        )
    else:
        def split_row(x):
            raw_split_result = split_datapoint(
                x['obs'],
                population_sizes.loc[x.name]*x[groups_to_split_into],
                rate_patterns.loc[x['pattern_id']],
                model=model,
                observed_total_se=x['obs_se']
            )
            return pd.Series(
                [
                    (estimate, se) for estimate, se in zip(raw_split_result[0], raw_split_result[1])
                ],
                index=groups_to_split_into)
        result_raw = (
            splitting_df
            .set_index('location_id')
            .apply(
                split_row,
                axis=1)
        )
        point_estimates = result_raw.applymap(lambda x: x[0])
        standard_errors = result_raw.applymap(lambda x: x[1])
        result = pd.concat([point_estimates, standard_errors], keys=[
                           'estimate', 'se'], axis=1)  # .groupby(level=0).sum()

    return result
