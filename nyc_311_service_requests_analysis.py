# %% [markdown]
# # Hello there!
# 
# I'm formatting, analyzing, and manipulating a notoriously messy and complicated dataset called *NYC 311 Service Requests*.
# 
# This data is updated daily and contains information about public service requests across NYC. The nature of the requests ranges wildly but is focused around non-emergency reports and inquiries submitted by the public such as missed home delivery meals, noise complaints, lost property, and litter.
# 
# Because the dataset is MASSIVE, I'm only working with data from 2025. If you'd like to see or export the same data I'm analyzing, visit [this link](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/data_preview) and adjust the filters to 2025. My row count at the time of downloading was **3,655,017**.
# 
# Let's get ready to analyze some data!

# %% [markdown]
# # What are we going to look at?
# 
# Because there is so much data presented to us, there are a multitude of different things we can try to determine from it all. In order to keep us on track, we're going to be answering 5 different questions...
# 
# 1. How fast do things get resolved? (resolution time analysis)
# 2. Who handles the complaints? (we will look at the agency breakdown)
# 3. What do people in NYC complain about? (this is complaint volume by type)
# 4. What happens where? (geographic patterns)
# 5. When are complaints filed? (time patterns such as weekday, time, or even yearly seasons)
# 
# But before we answer these questions, we need to do some setup and initial cleaning.

# %% [markdown]
# ## Setup: loading and preparing the data
# 
# Working with a 3.6M row CSV is slow. To avoid waiting through that on every run, I...
# 
# 1. Load the CSV once.
# 2. Convert date columns to actual datetimes (instead of strings).
# 3. Force "object" columns into pandas's proper string type. This avoids subtle bugs and handles things like zip codes with leading zeros.
# 4. Save the cleaned DataFrame as a **Parquet file**, which is a fast, columnar format that preserves data types.
# 
# Every subsequent run loads from the Parquet file instead of re-parsing the CSV, which turns a 60+ second load into a couple of seconds.
# 
# ### Important 
# This notebook uses pandas and pyarrow. If you don't already have them installed, uncomment the `%pip install` line in the next cell.

# %%
# Required packages: pandas, pyarrow
# If you don't have them installed, uncomment and run the line below:
# %pip install pandas pyarrow matplotlib seaborn numpy

# %% [markdown]
# Note: this cell uses Parquet (a highly compressed file that is temporary storage of frequently accessed data) caching for speed. The first run takes around 60-90 seconds (loading and cleaning the CSV). After the Parquet file is created, runs load in 1-2 seconds from the cached Parquet file. If the cache ever needs to be rebuilt (e.g., the CSV was updated), set FORCE_REBUILD = True, run the cell once, then set it back to False.

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a clean visual style
sns.set_theme(style="whitegrid")

# %%
CSV_PATH = "311_Service_Requests_from_2020_to_Present_20260620.csv"
PARQUET_PATH = "311_2025.parquet"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Set to True to ignore the cached parquet and rebuild from CSV
FORCE_REBUILD = False

if FORCE_REBUILD and os.path.exists(PARQUET_PATH):
    print(f"Force rebuild requested - removing existing parquet")
    os.remove(PARQUET_PATH)

if os.path.exists(PARQUET_PATH):
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows from parquet (fast path)")
else:
    print("Loading CSV and converting...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['Created Date'] = pd.to_datetime(df['Created Date'], format=DATE_FORMAT, errors='coerce')
    df['Closed Date'] = pd.to_datetime(df['Closed Date'], format=DATE_FORMAT, errors='coerce')
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('string')
    
    df.to_parquet(PARQUET_PATH)
    print(f"Saved {len(df):,} rows to parquet for next time")

# %% [markdown]
# ## Sanity check: is this really only 2025?
# 
# I exported with a filter for 2025 in the NYC Open Data portal, but I want to verify that every single row really is from 2025 and not just the first and last ones.
# 
# The right check isn't "first and last rows look right" (that can miss rows from other years buried in the middle). The right check is to count how many rows are in each year. If only 2025 shows up, we're good to go.
# 

# %%
# Count records by year - should show only 2025
df['Created Date'].dt.year.value_counts()

# %% [markdown]
# ## What are we working with?
# 
# Let's see the structure of the dataset. We want to know how many rows, how many columns, what those columns are, and what their data types look like.

# %%
df.info()

# %% [markdown]
# ## What's with the columns?
# 
# After looking at the results, a couple of the columns have some abnormally long names. Later on when we create charts, these long names will make the chart labels long and ugly, so let's change a few now.
# 
# Note: For those in the future who may be viewing this, you may have to adjust the next cell to account for any column name changes.

# %%
# Rename the columns to something more readable.
df = df.rename(columns={
    'Problem (formerly Complaint Type)': 'Complaint Type',
    'Problem Detail (formerly Descriptor)': 'Descriptor'
})

# Ensure the changes took place.
df.info()

# %% [markdown]
# ## What do we really need?
# 
# Now that we know what all 44 columns are, we need to ask if we truly need all of these and the question is "no". While the data may be interesting to some, on the whole we really only need to focus some specific columns.
# 
# We'll mostly be working around about 10 columns as they give us the most interesting data stories. We may not end up using all of these in the notebook, but they're the ones I considered relevant.
# 
# #### Timestamps
# - Created Date
# - Closed Date
# 
# #### Complaints
# - Complaint Type
# - Descriptor
# 
# #### Agencies
# - Agency
# - Agency Name
# 
# #### Location
# - Borough
# 
# #### Outcomes
# - Status
# - Resolution Description
# 
# #### Record ID
# - Unique Key
# 
# These all give us enough data to focus on and will provide us the answers to our questions.

# %% [markdown]
# ## A quick helper
# 
# We'll use this pattern a lot which counts categorical values and shows percentages, so I'm defining it here to keep the rest of the notebook cleaner.

# %%
def count_and_pct(series, top_n=10):
    """Return a DataFrame with counts and percentages for a Series."""
    counts = series.value_counts().head(top_n)
    pcts = series.value_counts(normalize=True).head(top_n) * 100
    return pd.DataFrame({
        'Count': counts,
        'Percent': pcts.apply(lambda x: f"{x:.1f}%")
    })

# %% [markdown]
# # Data Quality Investigation
# 
# Before we can answer any real questions about this data, we need to understand what's actually in it. Real-world data is messy. There are usually records that are incomplete, contradictory, or just plain wrong. Let's identify the issues now so we can make informed decisions about how to handle them when we get into the analysis.

# %% [markdown]
# ## Something to consider about the dates...
# 
# Data can be messy, incomplete, or even corrupt. It would be great if all of the date cells had values, but that may not be the case. Before we go further, we should check a few things...
# 
# 1. Do all 'Created Date' and 'Closed Date' rows have a value? 
# 2. Do rows that have a 'Closed Date' value actually state that they are 'Closed' in the 'Status' column?
# 
# Let's do a quick sanity check.

# %%
print("'NaT' means 'Not a Time'")
print(f"Created Date NaT count: {df['Created Date'].isnull().sum():,}")
print(f"Closed Date NaT count: {df['Closed Date'].isnull().sum():,}")
print(f"'Closed' records without a closed date {df[df['Status'] == 'Closed']['Closed Date'].isnull().sum():,}")

ratio = df['Closed Date'].isnull().sum() / len(df)
print(f"Percentage not closed: {ratio:.1%}")

# %% [markdown]
# ## Wait a minute...
# 
# If you look at the results, every row has a created date, so that's good! However, we found that 66,118 records (about 1.8%) do not have a closed date AND 6,030 rows that are marked 'Closed' do not have a closed date.
# 
# Since this is a snapshot in real-time data, it's natural that there are a good number of requests that are still open. The real concern is why do we have data showing the request was closed but there is no closed date?
# 
# Maybe there is a commonality in these requests, so let's do a bit of investigating.

# %%
# Look at the contradictory records
closed_no_date = df[(df['Status'] == 'Closed') & (df['Closed Date'].isnull())]

# Are they concentrated in specific agencies?
print("\nBy Agency:")
print(count_and_pct(closed_no_date['Agency'], 10))

# By complaint type?
print("\nBy Complaint Type:")
print(count_and_pct(closed_no_date['Complaint Type'], 10))

# %% [markdown]
# ## The DHS pattern
# 
# Investigating the 6,030 'Closed without close date' records shows that 5,990 of them (over 99%) come from the Department of Homeless Services (DHS) on 'Homeless Person Assistance' requests. This isn't just random data. This indicates something real in how DHS handles these specific requests. Possible explanations include massive workflow closures, privacy considerations around sensitive cases, or a different definition of 'closed' for this complaint type. Going forward, I'll keep these records but exclude them from certain metrics later on, specifically resolution-time metrics. Anyone using this notebook for homelessness related analysis specifically should note that the time-to-resolution data is missing/incomplete for most DHS requests.

# %% [markdown]
# ## Setting up resolution_hours
# 
# To investigate the remaining data quality issues, we need a way to measure how long requests take to resolve. Let's add a calculated column for that now.
# 
# We're calculating in hours rather than days because requests that resolve in minutes or hours wouldn't be accurately described otherwise.
# 
# A note to remember: requests that are still open will display as 'NaT' in our new column. It's also worth noting that we'll likely see some negative numbers or even things that amount to a year+ in completion time. This is just another fun aspect of the messy data set, and we'll investigate both extremes next.

# %%
df['resolution_hours'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / 3600

# We'd rather not see scientific notation, so let's format to something easily readable
pd.set_option('display.float_format', '{:,.2f}'.format)
df['resolution_hours'].describe()

# %% [markdown]
# ## A lot going on
# 
# The median resolution time to resolve a request is 6.5 hours while 75% close just shy of 3 days (remember that the above is shown in hours). This is way faster than I expected a large city like NYC to resolve things.
# 
# The mean on the other hand is about 9 days, which is more than 30 times the median. This is more typical of what I would expect NYC to take to resolve issues, especially more involved ones.
# 
# Lastly, we have the 2 extremes which is the minimum that says -200 days and then the maximum which is about a year and a half. For the minimum, a complaint can't close before it's opened, so that's likely a data entry issue. As for the nearly 2 year long request, it's possible it's another data entry issue or just a very long request that's on-going.
# 
# Let's look into the negative resolution times first and see what's happening.

# %%
negative_resolution = df[(df['resolution_hours'] < 0)]

neg_count = len(negative_resolution)
total = len(df)
pct = neg_count / total * 100
print(f"Records with negative resolution: {neg_count:,} ({pct:.3f}% of all records)")

# Are they concentrated in specific agencies?
print("\nBy Agency:")
print(count_and_pct(negative_resolution['Agency'], 10))

# By complaint type?
print("\nBy Complaint Type:")
print(count_and_pct(negative_resolution['Complaint Type'], 10))

# %% [markdown]
# ## Another agency specific issue
# 
# From the results, we see that there are 914 records with a negative resolution (0.025% of total records). That's not really a large number considering 3.5 million + records, but it's worth noting down. 
# 
# The real intriguing data point here is that about 91% (833 of 914) are handled by DOT, and we can make an educated guess that the majority of those are 'Street Light Condition' complaints (824 records).
# 
# This is the second instance of a data quality issue being centralized to a specific agency. This makes me think that perhaps DOT has a specific bug in their system regarding reporting or a flawed process, but ultimately that's DOT knowledge.
# 
# Going forward, we will exclude these records from the resolution-time calculations considering a negative resolution time is impossible, but they'll still count towards complaint volume since the complaints are still valid.

# %% [markdown]
# ## Considering the other extreme
# 
# Since we investigated the negative resolution times, it's only natural to investigate the ones that are abnormally long, specifically over a year. Let's take a look at what's going on at the other extreme. Since we're getting more familiar with the data, let's make a prediction... My prediction is that the results will centralize around a specific agency and/or complaint, since that seems to be the trend. I think we'll see a similar number of rows to the negative resolution time, so maybe a few thousand. I also think that the complaints will revolve around high impact complaints such as building construction. 
# 
# Let's take a look.

# %%
# Records taking more than 1 year (8,760 hours) to resolve
year_plus = df[df['resolution_hours'] > 8760]

print(f"Records taking over 1 year: {len(year_plus):,} ({len(year_plus) / len(df) * 100:.3f}% of all records)")

print('\nBy Agency:')
print(count_and_pct(year_plus['Agency'], 10))

print('\nBy Complaint Type:')
print(count_and_pct(year_plus['Complaint Type'], 10))

# %% [markdown]
# ## Not just random noise
# 
# The results here prove my prediction to be mostly incorrect since we see a larger variety in complaint type and agencies involved. This makes me think that these are real issues and not some form of dirty data. Let's analyze what we found.
# 
# We see here that there are 1,103 records (0.030%) that took longer than a year to resolve. I was correct in that it was a similar number to the negative resolution issue, but the pattern here is different. When you look at the complaint types, it makes sense that these things took a long time to resolve. School maintenance, street light issues, damaged trees/sewers/sidewalks, building violations - these all take time to resolve. NYC is notorious for taking months to remove trees, school maintenance often waits for summer break, and building violations involve a legal process. 
# 
# It's also worth noting what the agencies are that make up the top percentages. These departments are physical infrastructure-heavy, so it makes sense that DPR (Department of Parks and Rec), DOB (Department of Buildings), DOE (Department of Education), and DOT (Department of Transportation) make up nearly 90% of the year+ resolution times. Additionally, we can match up the complaints as physical (roughly 54%), further adding to a notable pattern.
# 
# Going forward we'll include this data since it appears to be real complaints.

# %% [markdown]
# ## Data quality summary
# 
# Before moving into the actual analysis, let's quickly recap what we found and how we'll handle each issue:
# 
# 1. **DHS 'Closed without close date' records (6,030 records, ~99% DHS).** Real records, but excluded from resolution-time calculations since the close date is missing.
# 2. **Negative resolution times (914 records, ~91% DOT/Street Light).** Almost certainly a system bug. Excluded from resolution-time calculations.
# 3. **Resolution times over 1 year (1,103 records, spread across infrastructure agencies).** Likely real complaints that genuinely take a long time. Kept in the analysis.
# 
# Below is a quick summary of the numbers showing what's changed.
# 
# With those decisions made, we're ready to actually answer the questions.

# %%
# Copy old data set and create new data set with filtered out info. This is what we should be working with going forward.
df_original = df.copy()
df = df[
    (df['resolution_hours'] > 0) &
    (df['Status'] == 'Closed') &
    ~((df['Agency'] == 'DHS') & (df['Closed Date'].isnull()))
]

row_removal_difference = len(df_original) - len(df)
print(f'Rows removed: {row_removal_difference}')

print(f'Total rows in uncleaned data set: {len(df_original)}')
print(f'Total rows in cleaned data set: {len(df)}')

print(f'Percentage of rows removed: {(row_removal_difference / len(df_original) * 100):.2f}%')
print(f'Percentage of rows remaining: {(len(df) / len(df_original) * 100):.2f}%')

# %% [markdown]
# The results show that 134,517 (3.68%) total rows were removed. Let's take a little peak as to understand where those rows went.

# %%
# What status values did we lose?
print("Status distribution in REMOVED records:")
removed = df_original[~df_original.index.isin(df.index)]
print(removed['Status'].value_counts())
print(f"\nNegative resolution hours in original: {(df_original['resolution_hours'] < 0).sum():,}")
print(f"NaT resolution hours in original: {df_original['resolution_hours'].isnull().sum():,}")

# %% [markdown]
# # Breaking down the removed rows
# 
# Wait, why are we removing records we seemingly want?
# 
# There are 2 reasons for that:
# - **69,358 were technically 'Closed'** but weren't valid resolution times 
#   (missing close date, negative time, or simultaneous open/close)
# - **65,159 were still active** (In Progress, Open, Assigned, Pending, 
#   Started, or Unspecified) at the time of the data export
# 
# So the cleaned dataset isn't just "closed records" but
# "closed records with a valid, positive resolution time." That's the 
# data we can actually measure time-to-close on.

# %%
print(df['Status'].value_counts())
print(f"\nAny NaT resolution hours: {df['resolution_hours'].isnull().sum():,}")
print(f"Any negative resolution hours: {(df['resolution_hours'] < 0).sum():,}")
print(f"Any DHS records: {(df['Agency'] == 'DHS').sum():,}")

# %% [markdown]
# # Question 1: How fast do things get done?
# 
# Now that we've worked through the data quality issues and know what to trust and what to filter, we can actually answer the question. Let's look at the resolution times across all clean records and break it down by agency, complaint type, and borough.
# 
# Reminder: When I say what's valid and what to trust, we're talking about records with a closed status, positive hours (resolution time), and excluding the DHS records.

# %%
# Let's look at the fresh describe on the cleaned dataset.
df['resolution_hours'].describe()

# %% [markdown]
# # Resolution times on the cleaned dataset
# 
# After filtering to valid closed records with positive resolution times, we can see that the distribution is very similar to the original raw data. This is good! This indicates that what we removed didn't change the data that drastically. The median bumped up slightly from 6.5 hours to 7.15 and the mean bumped up as well from 211.3 hours to 214.7. The shape remains the same with everything being heavily right-skewed with most complaints being resolved within a day but a longer tail stretching to over a year.
# 
# If we notice the min of 0.00 hours, this simply reflects requests being opened and closed within minutes and does not indicate a data quality issue. It's completely viable for a request to be completed within the hour.
# 
# Let's take a look at some graphs.

# %%
fig, ax = plt.subplots(figsize=(12, 5))

# Define bins in log space instead of linear space
log_bins = np.logspace(
    np.log10(df['resolution_hours'].min() + 0.01),
    np.log10(df['resolution_hours'].max()),
    100
)

ax.hist(df['resolution_hours'],
        bins=log_bins,
        color='steelblue',
        edgecolor='none')

ax.set_xscale('log')

ax.set_title('Distribution of Resolution Times', fontsize=14, pad=15)
ax.set_xlabel('Resolution Time (hours, log scale)', fontsize=11)
ax.set_ylabel('Number of Complaints', fontsize=11)

# Readable tick labels
ax.set_xticks([1, 6, 24, 168, 720, 8760])
ax.set_xticklabels(['1hr', '6hrs', '1 day', '1 week', '1 month', '1 year'])

# Median line
median_val = df['resolution_hours'].median()
ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5,
           label=f'Median: {median_val:.1f} hrs')

ax.legend()
plt.tight_layout()
plt.show()

# %%
# These need to go in Question 4
print("By Borough:")
print(count_and_pct(df['Borough'], 6))

# This needs to go in Question 3
print("\nBy Complaint Type:")
print(count_and_pct(df['Complaint Type'], 10))


