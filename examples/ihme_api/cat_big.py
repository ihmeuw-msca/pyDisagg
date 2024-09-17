import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Assuming the CatSplitter and configuration classes have been imported correctly
from pydisagg.ihme.splitter import (
    CatSplitter,
    CatDataConfig,
    CatPatternConfig,
    CatPopulationConfig,
)

# Set a random seed for reproducibility
np.random.seed(42)

# Sizes to test
sizes = [100, 1000, 10000]
times = []

# List of possible location IDs
all_location_ids = np.arange(1000, 2000)

for size in sizes:
    print(f"\nProcessing size: {size}")
    # Generate study_ids
    study_ids = np.random.randint(1000, 9999, size=size)
    # For simplicity, set the year_id to 2010 for all rows
    year_ids = np.full(size, 2010)
    # Generate 'mean' and 'std_err'
    means = np.random.uniform(0.1, 0.5, size=size)
    std_errs = np.random.uniform(0.01, 0.05, size=size)
    # Generate 'location_id' lists
    location_ids = []
    for _ in range(size):
        # For each row, select between 1 and 5 random location IDs
        num_locations = np.random.randint(1, 6)
        loc_ids = np.random.choice(
            all_location_ids, size=num_locations, replace=False
        ).tolist()
        location_ids.append(loc_ids)
    # Create the pre_split DataFrame
    pre_split = pd.DataFrame(
        {
            "study_id": study_ids,
            "year_id": year_ids,
            "location_id": location_ids,
            "mean": means,
            "std_err": std_errs,
        }
    )

    # Flatten the list of location_ids to get all unique location IDs used
    unique_location_ids = set()
    for loc_list in location_ids:
        unique_location_ids.update(loc_list)
    unique_location_ids = list(unique_location_ids)

    # Pattern DataFrame for all location_ids
    data_pattern = pd.DataFrame(
        {
            "location_id": unique_location_ids,
            "year_id": np.full(len(unique_location_ids), 2010),
            "mean": np.random.uniform(0.1, 0.5, size=len(unique_location_ids)),
            "std_err": np.random.uniform(0.01, 0.05, size=len(unique_location_ids)),
        }
    )

    # Population DataFrame for all location_ids
    data_pop = pd.DataFrame(
        {
            "location_id": unique_location_ids,
            "year_id": np.full(len(unique_location_ids), 2010),
            "population": np.random.randint(
                10000, 1000000, size=len(unique_location_ids)
            ),
        }
    )

    # Configurations
    data_config = CatDataConfig(
        index=["study_id", "year_id"],
        target="location_id",
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
        data=data_config,
        pattern=pattern_config,
        population=population_config,
    )

    # Perform the split and time it
    start_time = time.time()

    final_split_df = splitter.split(
        data=pre_split,
        pattern=data_pattern,
        population=data_pop,
        model="rate",
        output_type="rate",
        n_jobs=-1,  # Use all available cores
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    print(f"Size: {size}, Time taken: {elapsed_time:.2f} seconds")

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(sizes, times, marker="o")
plt.xlabel("Number of Rows in Data")
plt.ylabel("Time Taken (seconds)")
plt.title("Runtime vs Data Size for CatSplitter")
plt.grid(True)
plt.show()
