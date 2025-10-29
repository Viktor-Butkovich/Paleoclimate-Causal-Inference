from typing import List
import polars as pl
import netCDF4 as nc
import numpy as np


def get_year_bin(year: int) -> int:
    if year < -800000:  # Nearest 50,000
        return round(year / 50000) * 50000
    elif year < -20000:  # Nearest 2000
        return round(year / 2000) * 2000
    elif year < 0:  # Nearest 250
        return round(year / 250) * 250
    elif year < 1850:  # Nearest 50
        return round(year / 50) * 50
    elif year < 2025:  # Nearest 1
        return round(year)
    else:  # Next 1000
        return round((year + 1000) / 1000) * 1000


def include_sample(sample) -> bool:
    if sample.get("paleoData_units") != "degC":
        return False
    elif sample.get("paleoData_useInGlobalTemperatureAnalysis", "TRUE") == "FALSE":
        return False
    elif "DELETE" in sample.get("paleoData_QCnotes", ""):
        return False
    elif sample.get("age", None) == None:
        return False
    elif sample.get("paleoData_values")[0] == "nan" or sample.get("age")[0] == "nan":
        return False
    else:
        return True


def year_bins_transform(df: pl.DataFrame, valid_year_bins: List[int]) -> pl.DataFrame:
    df = (
        df.with_columns(
            pl.col("year")
            .map_elements(get_year_bin, return_dtype=pl.Int64)
            .alias("year_bin")
        )
        .group_by("year_bin")
        .agg([pl.col(col).mean().alias(col) for col in df.columns if col != "year_bin"])
        .drop("year")
    )
    missing_year_bins = set(valid_year_bins) - set(df["year_bin"].unique())
    missing_entries = pl.DataFrame(
        {
            "year_bin": list(missing_year_bins),
            **{
                col: [None] * len(missing_year_bins)
                for col in df.columns
                if col != "year_bin"
            },
        }
    )

    df = pl.concat([df, missing_entries]).sort("year_bin")

    for col in df.columns:
        if col != "year_bin":
            df = df.with_columns(pl.col(col).interpolate().alias(col))

    df = df.with_columns(
        [
            pl.col(col).fill_null(strategy="backward").fill_null(strategy="forward")
            for col in df.columns
        ]
    )

    for col in [
        "co2_ppm",
        "co2_radiative_forcing",
        "anomaly",
        "be_ppm",
        "VADM",
    ]:  # Remove future filled null values
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col("year_bin") > 2025)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df


def get_climate_month(modern_temperature_grid: nc.Dataset, lat, lon, month_idx):
    """
    Gets the average temperature for the nearest latitude and longitude.

    Parameters:
    dataset (Dataset): The netCDF4 dataset.
    lat (float): The latitude.
    lon (float): The longitude.

    Returns:
    float: The average temperature.
    """
    # The temperature variable holds anomaly delta degC for each month since 1850, while the climatology variable holds the historical average degC for each month
    return float(
        modern_temperature_grid.variables["climatology"][
            :, round(lat) + 89, round(lon) + 179
        ][month_idx]
    )


def get_climate(modern_temperature_grid: nc.Dataset, lat, lon):
    # The temperature variable holds anomaly delta degC for each month since 1850, while the climatology variable holds the historical average degC for each month
    return float(
        np.mean(
            modern_temperature_grid.variables["climatology"][
                :, round(lat) + 89, round(lon) + 179
            ]
        )
    )


def get_weighted_global_average_climate(modern_temperature_grid: nc.Dataset):
    # Latitudes and longitudes are not proportional to surface area.
    # To get a more accurate representation of the Earth's surface, we need to account for the cosine of the latitude.
    # This is because the distance between lines of longitude decreases as you move towards the poles.
    latitudes = np.arange(-90, 91, 1)
    weights = np.cos(np.radians(latitudes))
    weighted_sum = 0
    total_weight = 0
    for lat, weight in zip(latitudes, weights):
        for lon in range(-180, 181, 1):
            weighted_sum += get_climate(modern_temperature_grid, lat, lon) * weight
            total_weight += weight
    return weighted_sum / total_weight


def get_anomaly_when_month(
    modern_temperature_grid: nc.Dataset, lat, lon, year, month_idx
):
    temperature = modern_temperature_grid.variables["temperature"][
        (year - 1850) * 12 + month_idx, lat, lon
    ]
    if (
        type(temperature) == np.ma.core.MaskedConstant
    ):  # Replace missing value with next year]
        return get_anomaly_when_month(
            modern_temperature_grid, lat, lon, year + 10, month_idx
        )
    else:
        return float(temperature)


def get_anomaly_when(
    modern_temperature_grid: nc.Dataset, lat, lon, year, month_idx=None
):
    if month_idx == None:
        month_time_series = [
            get_anomaly_when_month(modern_temperature_grid, lat, lon, year, month_idx)
            for month_idx in range(12)
        ]
        return float(np.mean(month_time_series))
    else:
        return get_anomaly_when_month(
            modern_temperature_grid, lat, lon, year, month_idx
        )


def calculate_solar_modulation(Q, M):
    """
    Calculate the solar modulation potential based on the given formula.
    Q = (1 / (a + b * phi)) * (c + M) / (d = eM)
    a, b, c, d, and e are known constants
        a = 0.7476
        b = 0.2458
        c = 2.347
        d = 1.077
        e = 2.274
    Q is the production rate of beryllium-10
    M is the geomagnetic field strength
    phi is the solar modulation, which we want to solve for
    By https://onlinelibrary.wiley.com/doi/full/10.1155/2014/345482?msockid=00f9caf6c69469371ab8dbfbc73c68e1:
        This formula relates beryllium-10 production rate with geomagnetic field strength and solar modulation potential
        Solar modulation potential is how much the sun reduces the intensity of cosmic rays reaching the Earth
            More intense cosmic rays cause more clouds, decreasing temperature, so high solar modulation potential means less clouds
            The inverse of solar modulation potential is an effective temperature predictor
    """
    a = 0.7476
    b = 0.2458
    c = 2.347
    d = 1.077
    e = 2.274
    if Q is None or M is None:
        return None
    phi = (1 / (a + b * ((c + M) / (d + e * M)) / Q)) - a
    return phi
