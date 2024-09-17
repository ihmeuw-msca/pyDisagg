import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import os
import gc
import matplotlib.ticker as ticker

from pydisagg.ihme.splitter import (
    CatSplitter,
    CatDataConfig,
    CatPatternConfig,
    CatPopulationConfig,
)

np.random.seed(42)

# Sizes to test
# sizes = [100, 1000, 10000, 100000]
sizes = [100, 1000, 10000, 100000, 250000, 500000, 750000, 1000000]
times_parallel = []
times_groupby = []
memory_parallel = []
memory_groupby = []


# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # in bytes
    return mem / (1024 * 1024)  # Convert to MB


# List of possible location IDs
all_location_ids = np.arange(1000, 2000)  # 1000 unique location IDs

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

    # Record memory before splitting
    mem_before = get_memory_usage()
    print(f"Memory Usage Before Splitting: {mem_before:.2f} MB")

    # Perform the split using parallel processing and time it
    start_time = time.time()

    final_split_df_parallel = splitter.split(
        data=pre_split,
        pattern=data_pattern,
        population=data_pop,
        model="rate",
        output_type="rate",
        n_jobs=-1,  # Use all available cores
        use_parallel=True,
    )

    end_time = time.time()
    elapsed_time_parallel = end_time - start_time
    times_parallel.append(elapsed_time_parallel)
    mem_after_parallel = get_memory_usage()
    memory_parallel.append(mem_after_parallel)
    print(f"Parallel - Size: {size}, Time taken: {elapsed_time_parallel:.2f} seconds")
    print(f"Memory Usage After Parallel Split: {mem_after_parallel:.2f} MB")

    # Perform garbage collection to free memory before next method
    del final_split_df_parallel
    gc.collect()

    # Perform the split using groupby (sequential processing) and time it
    start_time = time.time()

    final_split_df_groupby = splitter.split(
        data=pre_split,
        pattern=data_pattern,
        population=data_pop,
        model="rate",
        output_type="rate",
        use_parallel=False,
    )

    end_time = time.time()
    elapsed_time_groupby = end_time - start_time
    times_groupby.append(elapsed_time_groupby)
    mem_after_groupby = get_memory_usage()
    memory_groupby.append(mem_after_groupby)
    print(f"GroupBy  - Size: {size}, Time taken: {elapsed_time_groupby:.2f} seconds")
    print(f"Memory Usage After GroupBy Split: {mem_after_groupby:.2f} MB")

    # Clean up
    del final_split_df_groupby
    gc.collect()

    # Additional garbage collection between tests
    del pre_split
    del data_pattern
    del data_pop
    del splitter
    del data_config, pattern_config, population_config
    del study_ids, year_ids, means, std_errs, location_ids, unique_location_ids
    gc.collect()
    print(f"Memory Usage After Cleanup: {get_memory_usage():.2f} MB")

# Define colors for the plots
color_time_parallel = "#1f77b4"  # blue
color_time_groupby = "#aec7e8"  # light blue
color_mem_parallel = "#ff7f0e"  # orange
color_mem_groupby = "#ffbb78"  # light orange

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot time on the left y-axis
ax1.set_xlabel("Number of Rows in Data")
ax1.set_ylabel("Time Taken (seconds)", color="blue")
ax1.plot(
    sizes,
    times_parallel,
    marker="o",
    label="Time - Parallel",
    color=color_time_parallel,
)
ax1.plot(
    sizes, times_groupby, marker="s", label="Time - GroupBy", color=color_time_groupby
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.tick_params(axis="y", labelcolor="blue")

# Set x-axis ticks to show the sizes
ax1.set_xticks(sizes)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(ticker.NullFormatter())

# Create a twin Axes sharing the x-axis for memory usage
ax2 = ax1.twinx()
ax2.set_ylabel("Memory Usage (MB)", color="orange")
ax2.plot(
    sizes,
    memory_parallel,
    marker="o",
    label="Memory - Parallel",
    color=color_mem_parallel,
)
ax2.plot(
    sizes, memory_groupby, marker="s", label="Memory - GroupBy", color=color_mem_groupby
)
ax2.set_xscale("log")
ax2.tick_params(axis="y", labelcolor="orange")

# Add grid
ax1.grid(True, which="both", ls="--")

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.title("Time and Memory Usage Comparison: Parallel vs GroupBy in CatSplitter")
plt.show()
