import pandas as pd
import numpy as np

# Import CatSplitter and configurations from your package
from pydisagg.ihme.splitter import (
    CatSplitter,
    CatDataConfig,
    CatPatternConfig,
    CatPopulationConfig,
)

# Set a random seed for reproducibility
np.random.seed(42)

# -------------------------------
# Example DataFrames
# -------------------------------

# Pre-split DataFrame with 3 rows: 1 for Iowa, 2 for Washington
pre_split = pd.DataFrame(
    {
        "study_id": np.random.randint(1000, 9999, size=3),  # Unique study IDs
        "state": ["IA", "WA", "WA"],
        "county": [
            ["Johnson", "Scott", "Cedar", "Polk", "Linn"],  # Iowa counties
            [
                "King",
                "Pierce",
                "Snohomish",
                "Spokane",
                "Clark",
            ],  # WA row 1 counties
            ["King", "Pierce"],  # WA row 2 counties
        ],
        "year_id": [2010, 2010, 2010],
        "mean": [0.2, 0.3, 0.3],
        "std_err": [0.01, 0.02, 0.02],
    }
)

# Create a list of all counties mentioned
all_counties = [
    "Johnson",
    "Scott",
    "Cedar",
    "Polk",
    "Linn",  # Iowa counties
    "King",
    "Pierce",
    "Snohomish",
    "Spokane",
    "Clark",  # WA row 1 counties
    "Thurston",
    "Yakima",  # WA row 2 counties
]

# Pattern DataFrame for all counties
data_pattern = pd.DataFrame(
    {
        "county": all_counties,
        "year_id": [2010] * len(all_counties),
        "mean": np.random.uniform(0.1, 0.5, len(all_counties)),
        "std_err": np.random.uniform(0.01, 0.05, len(all_counties)),
    }
)

# Population DataFrame for all counties
data_pop = pd.DataFrame(
    {
        "county": all_counties,
        "year_id": [2010] * len(all_counties),
        "population": np.random.randint(10000, 1000000, len(all_counties)),
    }
)

# -------------------------------
# Configurations
# -------------------------------

data_config = CatDataConfig(
    index=["study_id", "state", "year_id"],  # Include study_id in the index
    target="state",
    sub_target="county",
    val="mean",
    val_sd="std_err",
)

pattern_config = CatPatternConfig(
    index=["year_id"],  # Include 'state' if patterns differ by state
    sub_target="county",
    val="mean",
    val_sd="std_err",
)

population_config = CatPopulationConfig(
    index=["year_id"],  # Include 'state' if populations differ by state
    sub_target="county",
    val="population",
)

# Initialize the CatSplitter
splitter = CatSplitter(
    data=data_config, pattern=pattern_config, population=population_config
)

# Perform the split
try:
    final_split_df = splitter.split(
        data=pre_split,
        pattern=data_pattern,
        population=data_pop,
        model="rate",
        output_type="rate",
    )
    final_split_df.sort_values(by=["state", "study_id", "county"], inplace=True)
    print("\nFinal Split DataFrame:")
    print(final_split_df)
except ValueError as e:
    print(f"Error: {e}")
