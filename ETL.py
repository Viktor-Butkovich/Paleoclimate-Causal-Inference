# %%
import requests
import os

# %%

url = "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/climate12k/temperature/version1.0.0/Temp12k_v1_0_0.pkl"
# Sourced from https://www.ncei.noaa.gov/access/paleo-search/study/27330

local_filename = "Data/Temp12k_v1_0_0.pkl"


def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {filename}")


download_file(url, local_filename)

# %%
