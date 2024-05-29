==========
Quickstart
==========

Python
======

Import
~~~~~~
.. code-block:: python

    from pydisagg.ihme.splitter import (
        DataConfig,
        PatternConfig,
        PopulationConfig,
        AgeSplitter,
    )

Configuration
~~~~~~~~~~~~~
.. code-block:: python

    fpg_data_con = DataConfig(
        index=["unique_id", "location_id", "year_id", "sex_id"],
        age_lwr="age_start",
        age_upr="age_end",
        val="mean",
        val_sd="SE",
    )

    draw_cols = patterns_fpg.filter(regex="^draw_").columns.tolist()

    fpg_pattern_con = PatternConfig(
        by=["location_id", "year_id", "sex_id"],
        age_key="age_group_id",
        age_lwr="age_group_years_start",
        age_upr="age_group_years_end",
        draws=draw_cols,
        val="mean_draw",
        val_sd="var_draw",
    )

    fpg_pop_con = PopulationConfig(
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
        model="rate",
        output_type="rate",
    )


R
=

Import
~~~~~~
.. code-block:: r

    # R code goes here

Configuration
~~~~~~~~~~~~~
.. code-block:: r

    # R code goes here

Age Splitting
~~~~~~~~~~~~~
.. code-block:: r

    # R code goes here


IHME
====

Import
~~~~~~
.. code-block:: r

    #If you're using the SciComp beta image with R 4.4.0 then load the reticulate library like this:
    #library(reticulate)
 
    #If you are using the default image (R 4.2.2) then you need to load the reticulate library like this:
    library(reticulate, lib.loc = "/ihme/code/msca/miniconda3/envs/pyDisagg/lib/R/library/")
    reticulate::use_python("/ihme/code/msca/miniconda3/envs/pyDisagg/bin/python")
    splitter <- import("pydisagg.ihme.splitter")


