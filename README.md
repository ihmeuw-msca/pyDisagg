## Dissaggregation under Generalized Proportionality Assumptions
This package dissaggregates an estimated count observation into buckets based on the assumption that the rate (in a suitably transformed space) is proportional to some baseline rate. 

The most basic functionality is to perform disaggregation under the rate multiplicative model that is currently in use. 

The setup is as follows: 

Let $D_{1,...,k}$ be an aggregated measurement across groups ${g_1,...,g_k}$, where the population of each is $p_i,...,p_k$. Let $f_1,...,f_k$ be the baseline pattern of the rates across groups, which could have potentially been estimated on a larger dataset or a population in which have higher quality data on. Using this data, we generate estimates for $D_i$, the number of events in group $g_i$ and $\hat{f_{i}}$, the rate in each group in the population of interest by combining $D_{1,...,k}$ with $f_1,...,f_k$ to make the estimates self consistent. 

Mathematically, in the simpler rate multiplicative model, we find $\beta$ such that 
$$D_{1,...,k} = \sum_{i=1}^{k}\hat{f}_i \cdot p_i $$
Where
$$\hat{f_i} = T^{-1}(\beta + T(f_i)) $$

This yields the estimates for the per-group event count, 

$$D_i = \hat f_i \cdot p_i $$
For the current models in use, T is just a logarithm, and this assumes that each rate is some constant muliplied by the overall rate pattern level. Allowing a more general transformation T, such as a log-odds transformation, assumes multiplicativity in the associated odds, rather than the rate, and can produce better estimates statistically (potentially being a more realistic assumption in some cases) and practically, restricting the estimated rates to lie within a reasonable interval. 

## Current Package Capabilities and Models
Currently, the multiplicative-in-rate model RateMultiplicativeModel with $T(x)=\log(x)$ and the Log Modified Odds model LMO_model(m) with $T(x)=\log(\frac{x}{1-x^{m}})$ are implemented. Note that the LMO_model with m=1 gives a multiplicative in odds model.

A useful (but slightly wrong) analogy is that the multiplicative-in-rate is to the multiplicative-in-odds model as ordinary least squares is to logistic regression in terms of the relationship between covariates and output (not in terms of anything like the likelihood)

Increasing m in the model LMO_model(m) gives results that are more similar to the multiplicative-in-rate model currently in use, while preserving the property that rate estimates are bounded by 1. 
