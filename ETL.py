# %%
# Imports

import requests
import os
import pickle as pkl
import polars as pl
import netCDF4 as nc
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
}
for data_source in data_sources.values():
    download_file(data_source["url"], data_source["path"])

# %%
# Load data files locally
temp12k = pkl.load(open(data_sources["temp12k"]["path"], "rb"))
modern_temperature_grid = nc.Dataset(data_sources["berkeley_earth"]["path"], mode="r")

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
