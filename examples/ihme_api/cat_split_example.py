import pandas as pd
import numpy as np

# Assuming the CatSplitter and configuration classes have been imported correctly
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

# Pre-split DataFrame with 3 rows
pre_split = pd.DataFrame(
    {
        "study_id": np.random.randint(1000, 9999, size=3),  # Unique study IDs
        "year_id": [2010, 2010, 2010],
        "location_id": [
            [1234, 1235, 1236],  # List of location_ids for row 1
            [2345, 2346, 2347],  # List of location_ids for row 2
            [3456],  # Single location_id for row 3 (no need to split)
        ],
        "mean": [0.2, 0.3, 0.4],
        "std_err": [0.01, 0.02, 0.03],
    }
)

# Create a list of all location_ids mentioned
all_location_ids = [
    1234,
    1235,
    1236,
    2345,
    2346,
    2347,
    3456,
    4567,  # Additional location_ids
    5678,
]

# Pattern DataFrame for all location_ids
data_pattern = pd.DataFrame(
    {
        "location_id": all_location_ids,
        "year_id": [2010] * len(all_location_ids),
        "mean": np.random.uniform(0.1, 0.5, len(all_location_ids)),
        "std_err": np.random.uniform(0.01, 0.05, len(all_location_ids)),
    }
)

# Population DataFrame for all location_ids
data_pop = pd.DataFrame(
    {
        "location_id": all_location_ids,
        "year_id": [2010] * len(all_location_ids),
        "population": np.random.randint(10000, 1000000, len(all_location_ids)),
    }
)

# Print the DataFrames
print("Pre-split DataFrame:")
print(pre_split)
print("\nPattern DataFrame:")
print(data_pattern)
print("\nPopulation DataFrame:")
print(data_pop)

# -------------------------------
# Configurations
# -------------------------------

data_config = CatDataConfig(
    index=["study_id", "year_id"],  # Include study_id in the index
    target="location_id",  # Column containing list of targets
    val="mean",
    val_sd="std_err",
)

pattern_config = CatPatternConfig(
    index=["year_id"],
    target="location_id",
    val="mean",
    val_sd="std_err",
)

population_config = CatPopulationConfig(
    index=["year_id"],
    target="location_id",
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
    final_split_df.sort_values(by=["study_id", "location_id"], inplace=True)
    print("\nFinal Split DataFrame:")
    print(final_split_df)
except ValueError as e:
    print(f"Error: {e}")
