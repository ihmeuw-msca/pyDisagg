from typing import Optional

import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from pydisagg.models import LMO_model, SplittingModel


def split_datapoint(
    measured_count: float,
    bucket_populations: NDArray,
    rate_pattern: NDArray,
    measured_count_se: Optional[float] = None,
    model: Optional[SplittingModel] = LMO_model(1),
    CI_method: Optional[str] = 'delta-wald'
):
    '''
    Disaggregates a datapoint using the model given as input.
    Defaults to assuming multiplicativity in the odds ratio

    If no measured_count_se is given, returns scalar point estimate
    If measured_count_se is given, then returns a tuple 
        (point_estimate,standard_error,(CI_lower,CI_upper))
    '''
    return model.split_groups(
        bucket_populations,
        measured_count,
        measured_count_se,
        rate_pattern,
        CI_method=CI_method
    )


def split_dataframe(
    groups_to_split_into: list,
    observation_group_membership_df: DataFrame,
    population_sizes: DataFrame,
    baseline_patterns: DataFrame,
    use_se: Optional[bool] = False,
    model: Optional[SplittingModel] = LMO_model(1),
):
    '''
    Disaggregates datapoints and pivots observations into estimates for each group per pop id

    groups_to_split_into: list of groups to disaggregate observations into

    observation_group_membership_df: dataframe with columns location_id, pattern_id, obs, 
        and columns for each of the groups_to_split_into
        with dummy variables that represent whether or not 
        each group is included in the observations for that row. 
        This also optionally contains a obs_se column which will be used if use_se is True
        location_id represents the population that the observation comes from
        pattern_id gives the baseline that should be used for splitting

    population_sizes: dataframe with location_id as the index containing the 
        size of each group within each population (given the location_id)

    rate_patterns: dataframe with pattern_id as the index, and columns 
        for each of the groups_to_split where the entries represent the rate pattern in the given group to use for pydisagg. 

    use_se: Boolean, whether or not to report standard errors along with estimates
        if set to True, then observation_group_membership_df must have an obs_se column
    '''
    splitting_df = observation_group_membership_df.copy()
    if use_se == False:
        def split_row(x):
            return split_datapoint(
                x['obs'],
                population_sizes.loc[x.name]*x[groups_to_split_into],
                baseline_patterns.loc[x['pattern_id']],
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
                baseline_patterns.loc[x['pattern_id']],
                model=model,
                measured_count_se=x['obs_se']
            )
            return pd.Series(
                [
                    (estimate, se) for estimate, se in zip(raw_split_result[0], raw_split_result[1])],
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
