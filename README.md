# Setup Instructions

To create and activate a Python virtual environment and install dependencies, follow these steps:

## Windows

1. **Create the virtual environment:**
   ```
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   ```
   .\venv\Scripts\activate
   ```
3. **Install dependencies: (only after activating venv)**
   ```
   pip install -r requirements.txt
   ```

## macOS/Linux

1. **Create the virtual environment:**
   ```
   python3 -m venv venv
   ```
2. **Activate the virtual environment:**
   ```
   source venv/bin/activate
   ```
3. **Install dependencies: (only after activating venv)**
   ```
   pip install -r requirements.txt
   ```

---

After initial setup, just activate the virtual 
environment and update dependencies as needed.

See preprocessed dataset in `Outputs/anomaly.csv` and raw dataset in `Outputs/visualization_view.csv`. These datasets may be created by activating the virtual Python environment and running the ETL script `ETL.py`, which extracts, transforms, and loads (exports to .csv) the data from the original sources.

# Data Dictionary:
* **year_bin**: Year AD, grouped into bins of 2k years (e.g. -4000, -2000, 0, 2000, etc.).
* **anomaly**: Degrees Celsius anomaly relative to Berkeley Earth's 1961-1990 baseline.
    * Extracted from the Temp12k dataset of temperature proxies for the past 12k years, Bereiter et al. (2015)'s Antarctic ice core data for the past 800k years, and Berkeley Earth's global grid of measured temperatures since 1750, compared against the 1961-1990 baseline for each sample's location.
    * Counter-intuitively, the Temp12k dataset provides temperature proxy records through the last 1.3M years. It has been compiled from 1319 distinct samples and many studies, each of which contains a time series of temperature proxy records at a specific location.
* **co2_ppm**: Atmospheric CO2 concentration in parts per million (ppm).
    * Extracted from Antarctic ice core samples for the past 800k years, as well as recent Mauna Loa Observatory measurements since 1959.
* **co2_radiative_forcing**: Calculated field derived from co2_ppm by the calculations of Myhre et al. (1998) - roughly linear relationship with temperature when climate sensitivity is held constant.
* Orbital parameters: Simulated by `https://biocycle.atmos.colostate.edu/shiny/Milankovitch/`, based on the calculations of J. Laskar et al (2004). These are broadly accepted as the main drivers of Earth's glacial cycles over the past several million years.
    * **eccentricity**: Eccentricity of the ellipse along which Earth orbits the Sun.
    * **obliquity**: Earth's axial tilt.
    * **perihelion**: Related to the precession of the equinoxes, which is related to distribution of sunlight by season.
    * **insolation**: Insolation at 65°N on the day of the summer solstice, a proxy for the summer warmth near glaciated regions of the Northern Hemisphere.
    * **global_insolation**: Average global insolation.
        * Notably, these values can be accurately simulated back and forward at least 1 million years.
   * **be_ppm**: Beryllium-10 concentration in parts per million (ppm) - Be-10 is produced by cosmic rays in the atmosphere, and so are a proxy for cosmic ray flux.
   * **VADM**: Virtual Axial Dipole Moment - A measure of Earth's magnetic field strength, which modulates cosmic ray flux.
   * **solar_modulation**: Measurement of solar activity derived from Be-10 concentration and VADM by the calculations of Marsh (2014).
       * Marsh theorizes that solar modulation is a core driver of glacial cycles.
       * Solar modulation -> cosmic ray flux <- VADM, and cosmic ray flux -> temperature, so solar modulation is the cleanest proxy for cosmic ray flux and its effect on temperature.

# Data Sources
Formal citations to be added later. See the `Manual/README.md` and the source links in ETL.py's `data_sources` dictionary for more information.

List of Temp12k samples and corresponding studies: Proof that Temp12k does in fact contain data beyond 12k years ago.
* https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/climate12k/temperature/version1.0.0/Temp12k_v1_essential_metadata.xlsx