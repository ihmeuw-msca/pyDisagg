==========
Quickstart
==========

Below are examples of how to load the package in Python and R. The Python example shows age-splitting, and the R example is for sex-splitting. However, both age and sex splitting are available in either Python or R.

Python
======

Import
~~~~~~
.. code-block:: python

    from pydisagg.ihme.splitter import (
        AgeDataConfig,
        AgePatternConfig,
        AgePopulationConfig,
        AgeSplitter,
    )

Configuration
~~~~~~~~~~~~~
.. code-block:: python

    fpg_data_con = AgeDataConfig(
        index=["unique_id", "location_id", "year_id", "sex_id"],
        age_lwr="age_start",
        age_upr="age_end",
        val="mean",
        val_sd="SE",
    )

    draw_cols = patterns_fpg.filter(regex="^draw_").columns.tolist()

    fpg_pattern_con = AgePatternConfig(
        by=["location_id", "year_id", "sex_id"],
        age_key="age_group_id",
        age_lwr="age_group_years_start",
        age_upr="age_group_years_end",
        draws=draw_cols,
        val="mean_draw",
        val_sd="var_draw",
    )

    fpg_pop_con = AgePopulationConfig(
        index=["age_group_id", "location_id", "year_id", "sex_id"],
        val="population",
    )

Age Splitting
~~~~~~~~~~~~~
.. code-block:: python

    age_splitter = AgeSplitter(
        data=fpg_data_con, pattern=fpg_pattern_con, population=fpg_pop_con
    )

    result = age_splitter.split(
        data=fpg_df,
        pattern=patterns_fpg,
        population=pops_df,
        model="logodds",
        output_type="rate",
    )


R
=

Import
~~~~~~
.. code-block:: r

    library(reticulate)
    reticulate::use_python("/some/path/to/miniconda3/envs/your-conda-env/bin/python")
    splitter <- import("pydisagg.ihme.splitter")

Configuration
~~~~~~~~~~~~~
.. code-block:: r

    sex_splitter = splitter$SexSplitter(
    data=splitter$SexDataConfig(
        index=c("nid","seq", "location_id", "year_id", "sex_id","age_lwr","age_upr"),
        val="val",
        val_sd="val_sd"
    ),
    pattern=splitter$SexPatternConfig(
        by=list('year_id'),
        val='draw_mean',
        val_sd='draw_sd'
    ),
    population=splitter$SexPopulationConfig(
        index=c('location_id', 'year_id'),
        sex="sex_id",
        sex_m=1,
        sex_f=2,
        val='population'
    )
)

Sex Splitting
~~~~~~~~~~~~~
.. code-block:: r

    result_sex_df = sex_splitter$split(data=pre_split,
                                   pattern= sex_pattern,population= sex_pop,
                                   model="rate"
                                   output_type = "total")


