"""Module containing high level api for splitting"""
from typing import Optional, Union, Literal

import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from pydisagg.models import LogOdds_model
from pydisagg.DisaggModel import DisaggModel


def split_datapoint(
    observed_total: float,
    bucket_populations: NDArray,
    rate_pattern: NDArray,
    observed_total_se: Optional[float] = None,
    model: Optional[DisaggModel] = LogOdds_model(),
    output_type: Literal['total','rate'] = 'total',
    CI_method: Optional[str] = 'delta-wald'
) -> Union[tuple,NDArray]:
    """Disaggregate a datapoint using the model given as input.
    Defaults to assuming multiplicativity in the odds ratio

    If output_type=='total', then this outputs estimates for the observed amount in each group
        such that the sum of the point estimates equals the original total
    If output_type=='rate', then this estimates rates for each group 
        (and doesn't multiply the rates out by the population)


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
    output_type: Literal['total','rate'], optional
        One of 'total' or 'rate'
        Type of splitting to perform, whether to disaggregate and return the estimated total
        in each group, or estimate the rate per population unit. 
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
    if output_type=='total':
        return model.split_groups(
            bucket_populations,
            observed_total,
            observed_total_se,
            rate_pattern,
            CI_method=CI_method
        )
    if output_type=='rate':
        return model.split_groups_rate(
            bucket_populations,
            observed_total,
            observed_total_se,
            rate_pattern,
            CI_method=CI_method
        )
    else:
        raise("ERROR:output_type must be one of either 'total' or 'rate'")


def split_dataframe(
    groups_to_split_into: list,
    observation_group_membership_df: DataFrame,
    population_sizes: DataFrame,
    rate_patterns: DataFrame,
    use_se: Optional[bool] = False,
    model: Optional[DisaggModel] = LogOdds_model(),
    output_type: Literal['total','rate'] = 'total',
    demographic_id_columns : Optional[list] = None,
) -> DataFrame:
    """Disaggregate datapoints and pivots observations into estimates for each group per demographic id

    If output_type=='total', then this outputs estimates for the observed amount in each group
        such that the sum of the point estimates equals the original total
    If output_type=='rate', then this estimates rates for each group 
        (and doesn't multiply the rates out by the population)

    Parameters
    ----------
    groups_to_split_into : list
        list of groups to disaggregate observations into
    observation_group_membership_df : DataFrame
        Dataframe with columns demographic_id, pattern_id, obs,
        and columns for each of the groups_to_split_into
        with dummy variables that represent whether or not
        each group is included in the observations for that row.
        This also optionally contains a obs_se column which will be used if use_se is True
        demographic_id represents the population that the observation comes from
        pattern_id gives the baseline that should be used for splitting
    population_sizes : DataFrame
        Dataframe with demographic_id as the index containing the
        size of each group within each population (given the demographic_id)
        INDEX FOR THIS DATAFRAME MUST BE DEMOGRAPHIC ID(PANDAS MULTIINDEX OK)
    rate_patterns : DataFrame
        dataframe with pattern_id as the index, and columns
        for each of the groups_to_split where the entries represent the rate pattern
        in the given group to use for pydisagg.
    use_se : Optional[bool], optional
        whether or not to report standard errors along with estimates
        if set to True, then observation_group_membership_df must have an obs_se column
        , by default False
    model : Optional[DisaggModel], optional
        DisaggModel to use for splitting, by default LogOdds_model()
    output_type: Literal['total','rate'], optional
        One of 'total' or 'rate'
        Type of splitting to perform, whether to disaggregate and return the estimated total
        in each group, or estimate the rate per population unit. 
    demographic_id_columns : Optional[list]
        Columns to use as demographic_id
        Defaults to None. If None is given, then we assume 
        that there is a already a demographic id column that matches the index in population_sizes. 
        Otherwise, we create a new demographic_id column, zipping the columns chosen into tuples

    Returns
    -------
    DataFrame
        Dataframe where each row corresponds to one of obs, with one or
            two columns for each of the groups_to_split_into, giving the estimate
        If use_se==True, then has a nested column indexing, where both the
            point estimate and standard error for the estimate for each group is given.
    """
    splitting_df = observation_group_membership_df.copy()
    if demographic_id_columns is not None:
        splitting_df['demographic_id']=list(
            zip(
                *[splitting_df[id_col] for id_col in demographic_id_columns]
            )
        )

    if use_se is False:
        def split_row(x):
            return split_datapoint(
                x['obs'],
                population_sizes.loc[x.name]*x[groups_to_split_into],
                rate_patterns.loc[x['pattern_id']],
                model=model,
                output_type=output_type
            )
        result = (
            splitting_df
            .set_index('demographic_id')
            .apply(
                split_row,
                axis=1)
            .reset_index()
        )
    else:
        def split_row(x):
            raw_split_result = split_datapoint(
                x['obs'],
                population_sizes.loc[x.name]*x[groups_to_split_into],
                rate_patterns.loc[x['pattern_id']],
                model=model,
                observed_total_se=x['obs_se'],
                output_type=output_type
            )
            return pd.Series(
                [
                    (estimate, se) for estimate, se in zip(raw_split_result[0], raw_split_result[1])
                ],
                index=groups_to_split_into)
        result_raw = (
            splitting_df
            .set_index('demographic_id')
            .apply(
                split_row,
                axis=1)
        )
        point_estimates = result_raw.applymap(lambda x: x[0])
        standard_errors = result_raw.applymap(lambda x: x[1])
        result = pd.concat([point_estimates, standard_errors], keys=[
                           'estimate', 'se'], axis=1).reset_index()  # .groupby(level=0).sum()
        
    if demographic_id_columns is not None:
        demographic_id_df=pd.DataFrame(
        dict(zip(demographic_id_columns, zip(*result['demographic_id']))),
        index=result.index
        )
        result=pd.concat([demographic_id_df,result],axis=1).drop('demographic_id',axis=1)

    return result
