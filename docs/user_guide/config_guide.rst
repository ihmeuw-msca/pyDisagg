Configuration guide
===========

Sex and Age Configuration
-------------------------

All of the following configurations expect column names or a list of column names.

Age Configuration
-----------------

The ``AgeDataConfig`` is used for age-related data configurations.

- **AgeData config** :
    - **index**: a list of columns that makes each row unique (e.g., seq) *and* contains columns like location, year, and sex for later merging.
    - **age_lwr**: lower bound of age for a given row.
    - **age_upr**: upper bound of age for a given row.
    - **val**: the value you want split.
    - **val_sd**: the standard error for the measure you want split.

- **Age Pattern config** :
    - **by**: a list of columns you want to match the data to (how specific is your pattern in addition to the age_key?).
    - **age_key**: age_group_id (at IHME).
    - **age_lwr**: lower bound of a given age_key (can merge "age_group_years_start/end" from get_age_metadata if needed).
    - **age_upr**: upper bound of a given age_key.
    - **draws**: a list of draw columns (e.g., "draw_1", ... "draw_n").
    - **val** & **val_sd**: alternative to draws if you have a mean_draw and standard error. Otherwise will become the column names calculated from the draws (e.g., "pat_mean" and "pat_standard_error").

- **Age Population config** :
    - **index**: a list of columns that makes each row unique and contains columns like location, year, and sex for merging.
    - **val**: the population.

Sex Configuration
-----------------

The ``SexDataConfig`` is used for sex-related data configurations and doesn't take the ``age_lwr``/``age_upr`` params due to current implementation limitations.

- **SexData config** :
    - **index**: a list of columns that makes each row unique (e.g., seq) and contains columns like location, year, and sex for later merging.
    - **val**: the value you want split.
    - **val_sd**: the standard error for the measure you want split.

- **Sex Pattern config** :
    - **by**: a list of columns you want to match the data to (how specific is your pattern?).
    - **draws**: a list of draw columns (e.g., "draw_1", ... "draw_n").
    - **val** & **val_sd**: alternative to draws if you have a mean_draw and standard error. Otherwise will become the column names calculated from the draws (e.g., "pat_mean" and "pat_standard_error").

- **Sex Population config** :
    - **index**: a list of columns that makes each row unique and contains columns like location, year, and sex for merging.
    - **sex**: the column name with the sex identifying values (ex. sex_id).
    - **sex_m**: the value for males in the sex column.
    - **sex_f**: the value for females in the sex column.
    - **val**: the population.

These configurations are then used to create "Agesplitter" and "Sexsplitter" objects which are the framework for the data you want split.

.. admonition:: Working in progress...
    :class: Attention

    Current topics

    * PEP8
    * naming
    * linter/formatter
    * docstrings
    * type hints