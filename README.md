# NYC 311 Service Requests — EDA

Exploratory analysis of NYC's 311 service request data using Python.

## What this is

An exploratory data analysis of about 3.6 million NYC 311 service
requests filed in 2025 using pandas. The notebook encompases going through 
loading and cleaning the data, identifying any data quality issues, and 
answering questions about resolution times, agency patterns, and complaint volumes.

## Setup

This notebook uses pandas and pyarrow:

    pip install pandas pyarrow

You'll also need the dataset itself. It's too large for GitHub, so go here:

1. Download from [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/data_preview)
2. Filter to 2025 in the portal before exporting
3. Save it as `311_Service_Requests_from_2020_to_Present_20260620.csv`
   (or update CSV_PATH at the top of the notebook)

First run takes 60-90 seconds to load and cache the CSV as parquet.
Subsequent runs load in 1-2 seconds from the cache.

## What's in here

- `nyc_311_service_requests_analysis.ipynb` — the main notebook
- `nyc_311_service_requests_analysis.py` — same content as a python file (for git diffs)

## Questions answered (or in progress)

1. How fast do things get resolved? (in progress)
2. Who handles the complaints? (todo)
3. What do people in NYC complain about? (todo)
4. What happens where? (todo)
5. When are complaints filed? (todo)

## Findings so far

Three data quality issues identified and documented:

1. ~6,000 records marked "Closed" with no close date — 99% from DHS
2. ~900 records with negative resolution time — 91% from DOT/Street Light
3. ~1,100 records with resolution time over 1 year — spread across
   infrastructure-heavy agencies (likely legitimate slow cases, not data
   quality bugs)

More to come as the analysis progresses!