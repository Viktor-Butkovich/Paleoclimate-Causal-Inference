# %%
# Imports

import requests
import os
import pickle as pkl
import polars as pl
import netCDF4 as nc
import numpy as np
from scipy.io import loadmat
from modules import util
import warnings

# %%
# Extract web data files
warnings.filterwarnings("ignore", category=UserWarning)


def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {filename}")


data_sources = {
    "temp12k": {
        "url": "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/climate12k/temperature/version1.0.0/Temp12k_v1_0_0.pkl",
        "path": "Data/Temp12k_v1_0_0.pkl",
        "source": "https://www.ncei.noaa.gov/access/paleo-search/study/27330",
    },
    "berkeley_earth": {
        "url": "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc",
        "path": "Data/Berkeley_Earth_Land_and_Ocean_LatLong1.nc",
        "source": "http://berkeleyearth.org/data/",
    },
    "ice_core_800k": {
        "url": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/antarctica2015co2composite-noaa.txt",
        "path": "Data/ice_core_800k_co2.txt",
        "source": "https://www.ncei.noaa.gov/access/paleo-search/study/17975",
    },
    "co2_trends": {
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv",
        "path": "Data/co2_annmean_mlo.csv",
        "source": "https://gml.noaa.gov/ccgg/trends/data.html",
    },
    "ecs_u1428": {
        "url": "http://www1.ncdc.noaa.gov/pub/data/paleo/contributions_by_author/anderson2018/anderson2018-u1428.txt",
        "path": "Data/anderson2018-u1428.txt",
        "source": "https://www.ncdc.noaa.gov/paleo/study/27892",
    },
    "ecs_u1429": {
        "url": "http://www1.ncdc.noaa.gov/pub/data/paleo/contributions_by_author/anderson2018/anderson2018-u1429.txt",
        "path": "Data/anderson2018-u1429.txt",
        "source": "https://www.ncdc.noaa.gov/paleo/study/27892",
    },
    "japansea_u1430": {
        "url": "http://www1.ncdc.noaa.gov/pub/data/paleo/contributions_by_author/anderson2019-c/anderson2019-u1430.txt",
        "path": "Data/anderson2019-u1430.txt",
        "source": "https://www.ncdc.noaa.gov/paleo/study/27911",
    },
    "guaymas_2006b": {
        "url": "https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/dean2006b/dean2006b.txt",
        "path": "Data/dean2006b.txt",
        "source": "https://www.ncdc.noaa.gov/paleo/study/16054",
    },
    "sint_2000": {
        "url": "https://github.com/kjg136/MLdipolePredictions/raw/main/Sint2000.mat",
        "path": "Data/Sint2000.mat",
        "source": "https://doi.org/10.1093/gji/ggac195",
    },
}
for data_source in data_sources.values():
    download_file(data_source["url"], data_source["path"])

# %%
# Load data files locally
temp12k = pkl.load(open(data_sources["temp12k"]["path"], "rb"))
modern_temperature_grid = nc.Dataset(data_sources["berkeley_earth"]["path"], mode="r")
ice_core_800k = pl.read_csv(
    data_sources["ice_core_800k"]["path"],
    separator="\t",
    comment_prefix="#",
)
co2_annmean_mlo = pl.read_csv(data_sources["co2_trends"]["path"], comment_prefix="#")
ecs_u1428 = pl.read_csv(
    data_sources["ecs_u1428"]["path"],
    separator="\t",
    comment_prefix="#",
    truncate_ragged_lines=True,
).select(["age_ka-BP", "Be_ppm"])
ecs_u1429 = pl.read_csv(
    data_sources["ecs_u1429"]["path"],
    separator="\t",
    comment_prefix="#",
    truncate_ragged_lines=True,
).select(["age_ka-BP", "Be_ppm"])
japansea_u1430 = pl.read_csv(
    data_sources["japansea_u1430"]["path"],
    separator="\t",
    comment_prefix="#",
    truncate_ragged_lines=True,
).select(["age_ka-BP", "Be_ppm"])
guaymas_2006b = pl.read_csv(
    data_sources["guaymas_2006b"]["path"],
    separator="\t",
    comment_prefix="#",
    truncate_ragged_lines=True,
)
sint_2000 = loadmat(data_sources["sint_2000"]["path"])

# %%
# Transform data into a tabular format
month_indexes = {
    "january": 0,
    "february": 1,
    "march": 2,
    "april": 3,
    "may": 4,
    "june": 5,
    "july": 6,
    "august": 7,
    "september": 8,
    "october": 9,
    "november": 10,
    "december": 11,
    "winter": 0,
    "spring": 3,
    "summer": 6,
    "fall": 9,
    "cold season": 0,
    "coldest month": 0,
    "coldest": 0,
    "mean summer": 6,
    "1; summer": 6,
    "1 (summer)": 6,
    "warmest month": 6,
    "warmest": 6,
}
temperature_data = []
units = set()
num_samples = 0
ages = []
outliers = []
rewritten_samples = []

for sample in temp12k["TS"]:
    units.add(sample.get("paleoData_units"))
    if util.include_sample(sample):
        if (
            sample.get("paleoData_interpretation")
            and sample["paleoData_interpretation"][0].get("seasonality", "").lower()
            in month_indexes
        ):
            month_idx = month_indexes[
                sample["paleoData_interpretation"][0]["seasonality"].lower()
            ]
        else:
            # Some samples labeled with month as habitatSeason, some labeled in interpretation
            month_idx = month_indexes.get(
                sample.get("paleoData_habitatSeason", None), None
            )
        lat, lon = round(sample.get("geo_meanLat")), round(sample.get("geo_meanLon"))

        num_samples += 1
        if month_idx == None:
            climate = util.get_climate(modern_temperature_grid, lat, lon)
        else:
            climate = util.get_climate_month(
                modern_temperature_grid, lat, lon, month_idx
            )

        anomalies = []
        most_recent_age = sample["age"][0]
        most_recent_temperature = sample["paleoData_values"][0]
        temperature_offset = 0
        if most_recent_age != "nan" and most_recent_temperature != "nan":
            if most_recent_age < -50:
                # If most recent age is in the future, adjust to be present
                #   Most samples use age as years before 1950, but some seem to use a different convention
                shift_forward = -50 - most_recent_age
                sample["age"] = [
                    age + shift_forward for age in sample["age"] if age != "nan"
                ]
                most_recent_age = sample["age"][0]
            most_recent_year = 1950 - int(most_recent_age)
            if most_recent_year > 0:
                if not "T anomalies" in sample.get("paleoData_description", ""):
                    if most_recent_year >= 1850:
                        if month_idx == None:
                            most_recent_anomaly = util.get_anomaly_when(
                                modern_temperature_grid, lat, lon, most_recent_year
                            )
                        else:
                            most_recent_anomaly = util.get_anomaly_when_month(
                                modern_temperature_grid,
                                lat,
                                lon,
                                most_recent_year,
                                month_idx,
                            )
                    else:
                        most_recent_anomaly = util.get_anomaly_when(
                            modern_temperature_grid, lat, lon, 1850
                        )
                    # Add most recent temperature to each data point, such that the most recent temperature is assumed to be the modern average
                    #   Thus, even if the most recent temperature is different from the modern average (local variation, error), we still get an accurate anomaly vs age
        for age, temperature in zip(sample["age"], sample["paleoData_values"]):
            # Convert age (years BP) to a date (assuming current year is 1950 for BP conversion)
            if age != "nan" and temperature != "nan":
                diff = float(temperature) - most_recent_temperature
                anomaly = (
                    diff + most_recent_anomaly
                )  # Total anomaly is difference from most recent + anomaly of most recent
                temperature = climate + anomaly
                temperature_data.append(
                    {
                        "sample_id": num_samples,
                        "year": round(
                            most_recent_year - (float(age) - most_recent_age)
                        ),
                        "degC": temperature,
                        "anomaly": anomaly,
                        "geo_meanLat": lat,
                        "geo_meanLon": lon,
                    }
                )
                ages.append(age)
                anomalies.append(temperature_data[-1]["anomaly"])
        if abs(temperature_data[-1]["anomaly"]) > 20:
            sample["anomaly"] = anomalies
            rewritten_samples.append(sample)

# %%
# Convert tabular temperature data to a Polars DataFrame
temperature_df = pl.DataFrame(temperature_data)
# Remove temperature outliers that are more than 1.5 IQR away
q1 = temperature_df["degC"].quantile(0.25)
q3 = temperature_df["degC"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

temperature_df = temperature_df.filter(
    (temperature_df["degC"] >= lower_bound) & (temperature_df["degC"] <= upper_bound)
)

# %%
# Add modern temperature anomalies
if not os.path.exists("Data/precomputed_modern_temperature.parquet"):
    array_data = modern_temperature_grid.variables["temperature"][:]
    # Get the dimensions
    months, latitudes, longitudes = array_data.shape

    # Create arrays for month, latitude, and longitude
    total_elements = months * latitudes * longitudes
    indices = np.arange(total_elements)

    # Calculate the original indices
    month_array = indices // (latitudes * longitudes)
    latitude_array = ((indices % (latitudes * longitudes)) // longitudes) - 89
    longitude_array = (indices % longitudes) - 179

    # Flatten the 3-dimensional array into a 1-dimensional array
    flattened_array = array_data.flatten()

    flattened_array: np.ma.masked_array = flattened_array

    filled_array = flattened_array.filled(np.nan)

    # Replace NaN values with the item at the index 1555200 indexes later
    nan_indices = np.where(np.isnan(filled_array))[0]
    for idx in reversed(nan_indices):
        filled_array[idx] = filled_array[idx + 1555200]

    # Create a Polars DataFrame
    modern_temperature_df = pl.DataFrame(
        {
            "month": month_array,
            "geo_meanLat": latitude_array,
            "geo_meanLon": longitude_array,
            "anomaly": filled_array,
        }
    )
    # Remove odd latitudes and longitudes
    modern_temperature_df = modern_temperature_df.filter(
        (modern_temperature_df["geo_meanLat"] % 2 == 0)
        & (modern_temperature_df["geo_meanLon"] % 2 == 0)
    )

    # Group by latitude and longitude, and calculate the average anomaly for each group of 12 months
    modern_temperature_df = (
        modern_temperature_df.with_columns((1850 + pl.col("month") // 12).alias("year"))
        .group_by(["geo_meanLat", "geo_meanLon", "year"])
        .agg(pl.col("anomaly").mean().alias("anomaly"))
    )

    modern_temperature_grid.variables["climatology"].shape
    # Calculate the average annual temperature for each latitude and longitude combination
    climatology_data = modern_temperature_grid.variables["climatology"][:]
    # Get the dimensions
    months, latitudes, longitudes = climatology_data.shape

    # Calculate the average annual temperature
    average_climate = climatology_data.mean(axis=0)
    # Create arrays for latitude and longitude
    latitude_array = np.arange(latitudes) - 89
    longitude_array = np.arange(longitudes) - 179

    # Create a Polars DataFrame
    climatology_df = pl.DataFrame(
        {
            "geo_meanLat": np.repeat(latitude_array, longitudes),
            "geo_meanLon": np.tile(longitude_array, latitudes),
            "climate": average_climate.flatten(),
        }
    )
    # Remove odd latitudes and longitudes
    climatology_df = climatology_df.filter(
        (climatology_df["geo_meanLat"] % 2 == 0)
        & (climatology_df["geo_meanLon"] % 2 == 0)
    )

    # Merge climatology_df and grouped_df on latitude and longitude
    modern_temperature_df = climatology_df.join(
        modern_temperature_df, on=["geo_meanLat", "geo_meanLon"], how="inner"
    )
    modern_temperature_df = modern_temperature_df.with_columns(
        (pl.col("climate") + pl.col("anomaly")).alias("degC")
    ).drop("climate")

    # Assign a unique sample_id to each unique pair of geo_meanLat and geo_meanLon
    unique_locations = modern_temperature_df.select(
        ["geo_meanLat", "geo_meanLon"]
    ).unique()
    unique_locations = unique_locations.with_row_index(name="sample_id")
    unique_locations = unique_locations.with_columns(
        (pl.col("sample_id") + num_samples + 1).alias("sample_id")
    )  # Add num_samples to each sample_id to avoid duplicates

    # Join the unique locations with the merged_df to assign sample_id
    modern_temperature_df = modern_temperature_df.join(
        unique_locations, on=["geo_meanLat", "geo_meanLon"], how="left"
    )

    modern_temperature_df = modern_temperature_df.with_columns(
        pl.col("sample_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int64),
        pl.col("degC").cast(pl.Float64),
        pl.col("anomaly").cast(pl.Float64),
        pl.col("geo_meanLat").cast(pl.Int64),
        pl.col("geo_meanLon").cast(pl.Int64),
    )
    modern_temperature_df.write_parquet("Data/precomputed_modern_temperature.parquet")
else:
    modern_temperature_df = pl.read_parquet(
        "Data/precomputed_modern_temperature.parquet"
    )
temperature_df = pl.concat(
    [
        temperature_df,
        modern_temperature_df.select(
            [
                "sample_id",
                "year",
                "degC",
                "anomaly",
                "geo_meanLat",
                "geo_meanLon",
            ]
        ),
    ]
).with_row_index(name="temperature_id")

# Assign measurements to year bins
#   Since we are combining measurements from many sources and times, we need to bin them to avoid missing values
temperature_df = temperature_df.with_columns(
    pl.col("year")
    .map_elements(util.get_year_bin, return_dtype=pl.Int64)
    .alias("year_bin")
).with_columns(pl.col("year_bin").alias("time_id"))
valid_year_bins = list(temperature_df["year_bin"].unique())

# %%
# Incorporate orbital simulation data (Milankovitch cycles)
orbital_df = pl.read_csv("Manual/milankovitch_simulation.csv")
orbital_df = orbital_df.rename({"global.insolation": "global_insolation"})
orbital_df = util.year_bins_transform(orbital_df, valid_year_bins)
valid_year_bins += list(orbital_df["year_bin"].unique())
valid_year_bins = sorted(
    set(valid_year_bins)
)  # Add future simulated values to valid year bins

# %%
# Incorporate 800k year ice core CO2 data
co2_df = (
    ice_core_800k.with_columns((1950 - pl.col("age_gas_calBP")).alias("year"))
).drop("age_gas_calBP", "co2_1s_ppm")
co2_df = util.year_bins_transform(co2_df, valid_year_bins)

# %%
# Incorporate CO2 data since 1959
co2_df = pl.concat(
    [
        co2_df,
        co2_annmean_mlo.with_columns(
            pl.col("year").alias("year_bin"),
            pl.col("mean").alias("co2_ppm"),
        ).select(["year_bin", "co2_ppm"]),
    ]
)
# Include radiative forcing calculation
initial_co2_ppm = 228  # Pre-industrial CO2 concentration
co2_df = co2_df.with_columns(
    pl.col("co2_ppm")
    .map_elements(lambda x: 5.35 * np.log(x / initial_co2_ppm), return_dtype=pl.Float64)
    .alias("co2_radiative_forcing")
)

# If there are any duplicate year bins, prioritize the modern measurements over the ice core measurements
co2_df = co2_df.group_by("year_bin").agg(
    pl.col("co2_ppm").last().alias("co2_ppm"),
    pl.col("co2_radiative_forcing").last().alias("co2_radiative_forcing"),
)

# %%
# Incorporate Beryllium-10 sediment data
be10_df = (
    pl.concat([ecs_u1428, ecs_u1429, japansea_u1430], how="vertical")
    .rename({"Be_ppm": "be_ppm"})
    .with_columns(((1950 - (pl.col("age_ka-BP") * 1000)).alias("year")))
    .filter(pl.col("year") >= min(valid_year_bins))
    .filter(pl.col("be_ppm") != 999999)  # Remove missing values
    .select(["year", "be_ppm"])
    .sort("year")
)
"""
The Dean 2006b dataset massively swings between different Be10 values and seems to use different units than the Anderson datasets.
be10_df = pl.concat(
    [
        be10_df,
        guaymas_2006b.with_columns(
            ((1950 - (pl.col("age_calkaBP") * 1000)).alias("year"))
        )
        .rename({"Be ppm": "be_ppm"})
        .select(["year", "be_ppm"]),
    ],
    how="vertical",
)
"""

# Assign year bins and aggregate Be10 concentrations
be10_df = util.year_bins_transform(be10_df, valid_year_bins)
# %%
# Incorporate VADM magnetic field strength
vadm_df = pl.DataFrame(
    {
        "year": ((sint_2000["t"] * 1000) + 3000).flatten().round().astype(int),
        "VADM": sint_2000["d"].flatten(),
    }
).filter(pl.col("year") >= min(valid_year_bins))
vadm_df = util.year_bins_transform(vadm_df, valid_year_bins).with_columns(
    pl.col("VADM").abs().alias("VADM")
)
cosmic_df = be10_df.join(vadm_df, on="year_bin", how="left")

# %%
# Calculate solar modulation from magnetic field strength and Be10 concentration
cosmic_df = cosmic_df.with_columns(
    pl.struct(["be_ppm", "VADM"])
    .map_elements(
        lambda row: util.calculate_solar_modulation(row["be_ppm"], row["VADM"]),
        return_dtype=pl.Float64,
    )
    .alias("solar_modulation")
)

# %%
# Aggregate temperature across locations by year bin
view = (
    temperature_df.group_by("year_bin")
    .agg(pl.col("anomaly").mean().alias("anomaly"))
    .sort("year_bin")
)

# Ensure all year bins are present (including future bins)
view = pl.DataFrame({"year_bin": valid_year_bins}).join(view, on="year_bin", how="left")

# %%
# Join supplementary data
view = (
    view.join(co2_df, on="year_bin", how="left")
    .with_columns(pl.col("co2_ppm").cast(pl.Float64))
    .join(orbital_df, on="year_bin", how="left")
    .join(cosmic_df, on="year_bin", how="left")
)

# %%
# Store raw data for maximum-resolution visualizations
visualization_view = util.round_columns(
    util.normalize(view, exclude=["year_bin", "anomaly", "co2_ppm"]),
    num_places=3,
    exclude=["year_bin"],
).sort("year_bin")
# %%
# Clean the data

# Aggregate data to have a constant frequency of 2000 years
view = (
    view.with_columns((pl.col("year_bin") // 2000 * 2000).alias("year_bin"))
    .group_by("year_bin")
    .agg(pl.all().mean())
).sort("year_bin")


# Apply a rolling mean to the solar_modulation column - helps with sparse data
view = view.with_columns(
    pl.col("solar_modulation")
    .rolling_mean(window_size=3, center=True)
    .alias("solar_modulation")
)

# Apply linear interpolation to fill null values in all past rows
interpolated = view.filter(pl.col("year_bin") < 2024).fill_nan(None).interpolate()
non_interpolated = view.filter(pl.col("year_bin") >= 2024)

# Ensure interpolated has the same data types as non_interpolated (interpolate and None values can interfere with types)
interpolated = interpolated.with_columns(
    [pl.col(col).cast(non_interpolated.schema[col]) for col in interpolated.columns]
)

view = pl.concat([interpolated, non_interpolated], how="vertical").sort("year_bin")

# Filter to only years with complete data
view = view.filter((pl.col("year_bin") >= -740000) & (pl.col("year_bin") <= 2024))

# Apply forward interpolation
view = view.fill_null(strategy="forward")

view = util.round_columns(view, num_places=3, exclude=["year_bin"])

# %%
# Export final datasets
visualization_view.write_csv(
    "Outputs/visualization_view.csv",
)
view.write_csv(
    "Outputs/anomaly.csv",
)
# %%
