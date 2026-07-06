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
# Note: this cell uses Parquet (a highly compressed file that is temporary storage of frequently accessed data) caching for speed. The first run takes around 60-90 seconds (loading and cleaning the CSV). After the Parquet file is created, runs load in 1-2 seconds from the cached Parquet file. Set FORCE_REBUILD = True if you need to rebuild the parquet cache from scratch (e.g. if the CSV was updated). Set it back to False after rebuilding.

# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
# Lastly, we have the 2 extremes which is the minimum that says -200 days and then the maximum which is about a year and a half. For the minimum, a complaint can't close before it's opened, so that's likely a data entry issue. As for the nearly 2 year long request, it's possible this is another data entry issue or just a very long request that's on-going.
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
# Understanding the removed records:
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

# %% [markdown]
# # Initial analysis
# 
# Upon generating the histogram, we see something more interesting than a 
# simple right skew... there are 2 distinct peaks. The first and largest 
# peak sits around the 1 hour mark, representing requests that get handled 
# very quickly. These are likely noise complaints or other routine requests 
# that agencies can action immediately.
# 
# There's then a dip around the 6 hour to 1 day range. My instinct is that 
# this reflects requests coming in at the end of the work day and sitting 
# overnight unhandled, but that's worth investigating further.
# 
# After that dip, we see a second elevated stretch with some jagged spikes 
# through the 1 day to 1 week range, likely representing more labor-intensive 
# requests that go through a formal process before closing. Then one final 
# notable spike around the 1 month mark before the tail tapers off.
# 
# What's making up these different peaks? Let's find out.

# %%
# What's in the first peak - complaints closing within the first 2 hours
first_peak = df[df['resolution_hours'] <= 2]

print(f"Records in first peak: {len(first_peak):,}")
print(f"\nBy Complaint Type:")
print(count_and_pct(first_peak['Complaint Type'], 10))
print(f"\nBy Agency:")
print(count_and_pct(first_peak['Agency'], 10))

# %% [markdown]
# # The first peak analysis
# 
# After pulling the information for the first peak, we notice a few things that are pretty interesting. Firstly, we notice that about 90% of requests closed in under 2 hours are handled by the NYPD. This is very fascinating because it leads me to believe the NYPD either have a shorter process for handling requests or they are just incredibly fast. Without knowing the inner workings of how the NYPD operate, the data supports that NYPD officers are able to close requests on the spot without going through a formal process. Regarding complaint type, the top complaint types involve illegal parking and noise complaints across several subcategories, which is something I expected to see for NYC with how congested it is. As we look at the other complaint reasons, it's easy to see why the NYPD is the dominating agency as most all of these would require a police officer to attend to the issue.

# %%
# What's in that second peak around 1 month?
second_peak = df[(df['resolution_hours'] >= 480) & 
                 (df['resolution_hours'] <= 1080)]

print(f"Records in second peak: {len(second_peak):,}")
print(f"\nBy Complaint Type:")
print(count_and_pct(second_peak['Complaint Type'], 10))
print(f"\nBy Agency:")
print(count_and_pct(second_peak['Agency'], 10))

# %% [markdown]
# # The second peak analysis
# 
# From our results for the second peak, we can see that HPD (Department of Housing Preservation and Development) leads the Agency charge with 71242 records (54%). Given the nature of the department, it's natural that things regarding housing take a long time. In this case, it seems like requests take around 30 days for the agency, which leads me to believe that there is some sort of formal process that the agency goes through that results in about a month's completion time. If we look at what the complaints are for that second peak, we can see that the complaints are things that cannot be resolved in an hour or so as unsanitary conditions, plumbing issues, and water leaks all take time to fix, especially in a large city like New York City.

# %% [markdown]
# # Connecting the two peaks
# 
# When looking at the two, the two peaks reveal something about how NYC city services work and that's some services (police responses) are designed for immediate reaction and resolution while others (housing and maintenance) are following a more scripted, regulated process that takes weeks to months. Seeing the two peaks isn't a data issue but an indication of the inner workings and design of New York City's service resolution.
# 

# %% [markdown]
# # Agency time analysis
# 
# Now that we've seen the overall resolution time, let's get a bit specific on some resolution time analysis. Let's first take a look at how the top 15 agencies (by volume) handle requests.

# %%
# Top 15 agencies by complaint volume
top_agencies = df['Agency'].value_counts().head(15).index

agency_median = (df[df['Agency'].isin(top_agencies)]
                 .groupby('Agency')['resolution_hours']
                 .median()
                 .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(agency_median.index, agency_median.values, color='steelblue')

# Add value labels on each bar
for bar, val in zip(bars, agency_median.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} hrs', va='center', fontsize=9)

ax.set_title('Median Resolution Time by Agency (Top 15 by Volume)', 
             fontsize=14, pad=15)
ax.set_xlabel('Median Resolution Hours', fontsize=11)
ax.set_ylabel('Agency', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# From the bar graph generated, we can see that NYPD has the fastest resolution time median with 1.3 hours. This likely reflects my earlier statement that the NYPD are able to close requests with a less formal process leading to faster times. The resolution times and corresponding agencies make sense as you move up the chart. It makes sense that OTI (Office of Technology and Innovation), for example, takes a longer time to resolve because of formal ticketing systems and the nature of technology requests taking more time. It also makes sense that DOE (Department of Education) takes longer as well because of school schedules and requests likely being completed in the summer months. 
# 
# We can also notice the natural groupings of agencies and their requests. Same-day resolutions go to NYPD and DHS, multi-day go to DEP and DOT, weeks/month long requests go to HPD and DPR, and lastly several month plus go to TLC and EDC. This just helps us understand the different agencies and the types of services within the city.
# 
# A small interesting data point here is that DHS, an agency which had quite a bit of data quality issues earlier, resolves issues in about 8 hours (close to 1 work day). 
# 
# Lastly, we can see the huge outlier of EDC median taking almost a year at 310 days. This too makes sense as EDC (Economic Development Corporation) handles very large-scale projects that simply take long periods of time to complete. All in all, agency resolution times correlate to the nature of the request, which just makes sense. 

# %%
# Top 15 complaint types by volume
top_complaints = df['Complaint Type'].value_counts().head(15).index

complaint_median = (df[df['Complaint Type'].isin(top_complaints)]
                    .groupby('Complaint Type')['resolution_hours']
                    .median()
                    .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(complaint_median.index, complaint_median.values, 
               color='steelblue')

for bar, val in zip(bars, complaint_median.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} hrs', va='center', fontsize=9)

ax.set_title('Median Resolution Time by Complaint Type (Top 15 by Volume)', 
             fontsize=14, pad=15)
ax.set_xlabel('Median Resolution Hours', fontsize=11)
ax.set_ylabel('Complaint Type', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# Not surprisingly, the fastest complaint types are almost entirely NYPD-handled requests, while the slowest are predominantly HPD housing complaints. This is all very consistent with what we found in the agency breakdown.
# 
# By looking at the median resolution time by complaint type, we can see that specific noise complaints resolve very quickly. We can also see that illegal parking,  blocked driveways, and abandoned vehicles (which likely correlate to each other as blocked driveways are often due to illegal parking) are all resolved in a similar, short window of time. Again, this links back to likely being NYPD requests which we've seen have very quick resolution times.
# 
# What I find interesting is that the generic noise complaints, which seems to be a catch-all for non-specific complaints, have a median time of 66.8 hours of resolution while specific noise complaints are the fastest. Perhaps this is due to being more generic which results in different agencies being involved or generic meaning it's more involved.
# 
# Another interesting aspect is that water system complaints are resolved relatively quickly with 9.9 hour median resolution times while heat/hot water has 39.4 hour resolution times. This makes me think that the former relates more to water leaks or possibly dirty water in the tap which would take priority as safe drinking water is paramount. 
# 
# Lastly, we can see at the top the slowest complaints to be resolved are things like dirty conditions, paint/plaster which likely is whole building paint jobs, and the slowest being plumbing and unsanitary conditions, which has a separate complaint type than "dirty conditions." These all make sense as those types of complaints take longer to resolve with more formal inspection processes and likely because checks/verifications need to be performed after the service is completed.

# %%
# Drop unspecified
borough_median = (df[df['Borough'] != 'Unspecified']
                  .groupby('Borough')['resolution_hours']
                  .median()
                  .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(borough_median.index, borough_median.values, 
               color='steelblue')

for bar, val in zip(bars, borough_median.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} hrs', va='center', fontsize=9)

ax.set_title('Median Resolution Time by Borough', fontsize=14, pad=15)
ax.set_xlabel('Median Resolution Hours', fontsize=11)
ax.set_ylabel('Borough', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# From our third bar graph, which focuses on the resolution time per borough, we can see that Queens has the fastest resolution time at 4.2 hours with Brooklyn being fairly close with 6.7 hours. One possible explanation for this is that these boroughs are the largest and share more land borders without any real water divide (aside from the Jamaica Bay area) which could potentially result in more manpower being distributed there because of the larger area to cover. 
# 
# Manhattan and the Bronx are pretty close to each other as well being at 10 hours and 11.2 hours respectively, and that too makes sense because they're closer to each other. These two boroughs have rivers separating them from both each other and the other boroughs, which means there are fewer entry points to them, which may lead to longer response times. 
# 
# Staten Island takes the longest at 16.4 hours. While on a map Staten Island is about the same size as Brooklyn (a little smaller), it's much further away from the other boroughs and has a larger body of water separating them. With fewer access points, agencies probably aren't able to respond as quickly.
# 
# It's worth noting that the spread of the resolution times across the boroughs is relatively small compared to earlier variations of resolutions. The data shows that while location has some impact, what the complaint is matters more than where the complaint happened.

# %% [markdown]
# # Question 2: Who handles the complaints?
# 
# We've slightly covered this already, but it's time for a more in-depth analysis on which agency handles the majority of the complaints. From what we've seen, the NYPD handles the majority of complaints, but let's see if that still holds true after we break it down.

# %%
# Complaint volume by agency (top 15)
agency_counts = df['Agency'].value_counts().head(15).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(agency_counts.index, agency_counts.values, color='steelblue')

# Add value labels on each bar
for bar, val in zip(bars, agency_counts.values):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

ax.set_title('Complaint Volume by Agency (Top 15)', fontsize=14, pad=15)
ax.set_xlabel('Number of Complaints', fontsize=11)
ax.set_ylabel('Agency', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Complaint by agency analysis
# 
# After generating our bar graph, we can confirm that NYPD is the dominating agency by a significant margin of just over 1.7 million. Considering the data set houses 3.6 million records, it's amazing that the NYPD takes care of roughly 49% of all the complaints (in the cleaned dataset). But from what we've been seeing with the data, it makes sense that the NYPD is handling the majority of the requests as noise and parking complaints are the most common. The agency that's second is the HPD and that too makes logical sense as they deal with housing. NYC has a very large population, so it's only natural that with that many people living there, the request count involving living conditions would be high. 
# 
# It's also something to point out how drastic the gap is between the NYPD and HPD. HPD handles less than half of what the NYPD does and the drop-off continues heavily from there.
# 
# In the middle we can see DOHMH (Health and Mental Hygiene) at 54,766 and DHS (Homeless Services) at 45,441. These two are fairly close in number and I could speculate that some of the requests work across both agencies as homelessness can create health/mental health issues.
# 
# At the end of the graph we see OTI (Technology and Innovation) with only 204 requests. That's such a low number in the grand scheme of things but perhaps this is because, as we saw earlier, OTI median requests take 355.4 hours for completion or maybe it's because most technology requests are completed via private business rather than government services.

# %% [markdown]
# # Question 3: What do people complain about?
# 
# Like question 2, we've covered this a little bit but now it's time for an isolated analysis. We saw earlier that illegal parking and residential and street/sidewalk noise complaints were the leaders for the first peak of the distribution of resolution times, but that doesn't mean we'll see them as leaders here. Let's find out.

# %%
# Complaint volume by Complaint Type (top 15)
complaint_counts  = df['Complaint Type'].value_counts().head(15).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(complaint_counts.index, complaint_counts.values, color='steelblue')

# Add value labels on each bar
for bar, val in zip(bars, complaint_counts .values):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

ax.set_title('Complaint Volume by Complaint Type (Top 15)', fontsize=14, pad=15)
ax.set_xlabel('Number of Complaints', fontsize=11)
ax.set_ylabel('Complaint Type', fontsize=11)

plt.tight_layout()
plt.show()

print("\nBy Complaint Type:")
print(count_and_pct(df['Complaint Type'], 15))


# %% [markdown]
# # Complaints by type analysis
# 
# It would seem the first peak from the histogram accurately reflects most of the top complaints here. In the lead, we have illegal parking with more than half a million complaints (577,248), followed fairly close by Noise - Residential (463,171) and HEAT/HOT WATER (314,378). Each of the top 3 have about a 100,000 difference between them, so it's safe to say there likely isn't a connection, but the top 2 are complaints we've seen resolved quickly while HEAT/HOT WATER takes almost 2 days.
# 
# However, looking deeper, we can see that all "Noise" complaints totaled across 4 subcategories is roughly 756,000 complaints while illegal parking is roughly 575,000, resulting in overall noise complaints as the largest complaint "type."
# 
# Something that is intriguing to see is that Noise - Street/Sidewalk and Blocked Driveway are almost identical with 173,033 and 172,721 respectively. It's possible that these complaints are sometimes related (a blocked driveway could create noise on a street), but the dataset doesn't have enough detail to confirm if individual complaints are related.
# 
# After that borderline tie, we see UNSANITARY CONDITION with 117,244 complaints and then a pretty steady and close decline of the numbers. At the bottom we have Noise and Encampment at 55,706 and 47,995 respectively which could potentially be another connection.
# 
# It's easy to see here that the top 2 complaints are handled by NYPD while HEAT/HOT WATER, the third most common, is HPD handled. This links back nicely to earlier results with those 2 agencies being leaders in the first and second peaks.

# %% [markdown]
# # Question 4: What happens where?
# 
# We've touched briefly on median resolutions by borough, but we haven't really explored what types of complaints are more common across the boroughs. Let's dive deeper into what's going on in each borough.

# %%
# Drop unspecified
borough_counts = (df[df['Borough'] != 'Unspecified']['Borough']
                  .value_counts()
                  .head(5)
                  .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(borough_counts .index, borough_counts .values, color='steelblue')

# Add value labels on each bar
for bar, val in zip(bars, borough_counts .values):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

ax.set_title('Complaint Volume by Borough', fontsize=14, pad=15)
ax.set_xlabel('Number of Complaints', fontsize=11)
ax.set_ylabel('Borough', fontsize=11)

plt.tight_layout()
plt.show()

# Percentage counts
print("By Borough:")
print(count_and_pct(df[df['Borough'] != 'Unspecified']['Borough'], 5))

# Top 3 complaint types per borough
top3_per_borough = (df[df['Borough'] != 'Unspecified']
                    .groupby('Borough')['Complaint Type']
                    .apply(lambda x: x.value_counts().head(3))
                    .reset_index())

print('\n\n')
top3_per_borough.columns = ['Borough', 'Complaint Type', 'Count']
print(top3_per_borough.to_string(index=False))

# %% [markdown]
# # The borough analysis
# 
# From our graph, we can see that Brooklyn takes a commanding lead of the total complaint volume at just over 1 million complaints, which is nearly 30% (29.6% to be exact) of the total complaints. Not far behind are both Queens and the Bronx at 841,817 (23.9%) and 817,093 (23.2%) complaints for their respective boroughs. These 2 boroughs being so similar in complaint count is intriguing to me since Queens is geographically much larger than the Bronx.
# 
# After the Queens and Bronx borderline tie is Manhattan at 690,455 complaints (19.6%) which, although the data has been supporting the other boroughs get more complaints, is surprising to me since everyone thinks that NYC = Manhattan. Since studying this data, it's been interesting to see that Manhattan isn't the leading force in a lot of areas.
# 
# Lastly we have Staten Island at only 128,978 complaints, which is only 3.7% of the total. While I expected Staten Island to have a smaller complaint count, it's surprising to see that they have so few in the grand scheme. The most likely explanation for this is because of the population difference from Staten Island to the rest of the boroughs, as Staten Island has roughly 500,000 residents compared to Brooklyn's 2.7 million.
# 
# Regarding what the top complaints are for each borough, it's interesting to see that the top 3 complaints for each are all from the top 3 of the overall complaint volume, except for 2. It's natural for us to see the top 3 overall complaint types being represented here in some fashion, but seeing Queens and Staten Island's third top complaint differing is interesting to see.
# 
# We also see two outliers to the pattern in the Bronx and Manhattan. While the other 3 boroughs have Illegal Parking as their top complaint, the Bronx has residential noise complaints as theirs, and its nearly double its second-place complaint. Manhattan, on the other hand, has HEAT/HOT WATER as their top complaint for their borough, which likely reflects the vast number of apartment buildings where the heating systems provide heat across entire floors of residents. 
# 
# Queens third most common complaint is a blocked driveway. While this isn't in the top 3 of the overall complaint volume, it still ranks 5th. The reason we likely see it here in Queens is because the borough is more residential than the others. This means more houses, apartments, condos, etc. so seeing more complaints about a blocked driveway makes sense.
# 
# Staten Island, on the other hand, has Missed Collection as its third highest complaint. Like Queens, Staten Island is also more residential, so there are more living spaces for trash and recycling to be picked up from. With that in mind, you would assume that a blocked driveway would be the third highest complaint like Queens, but here we see missed collections. Staten Island is geographically more distant from the others, so it's likely because more travel time has to be put in to get there.

# %% [markdown]
# # Question 5: When are complaints filed?
# 
# Now that we're at our last question of the notebook, let's take a look at when complaints are filed. Let's take a look at 4 different graphs to better understand different time groupings. 
# 
# Chart 1 - Hour of Day: Here I expect to see the afternoon with a higher count as most people are awake and going about daily life.
# Chart 2 - Day of Week: I expect to see the weekends with a higher number as more people are home.
# Chart 3 - Month of Year: For this graph, I expect to see the summer months and the winter months with higher counts as in summer, people are out exploring/doing activities while winter the cold would create more complaints.
# Chart 4 - A hypothesis test: Since we've seen that noise complaints are consistently totaled high in complaint counts, I'm curious to see if the other complaints are higher during the middle of the day when people aren't home and then for noise complaints to be higher at night when people are.

# %%
# Day name mapping since days of the week go from 0 - 6
day_names = ['Monday', 'Tuesday', 'Wednesday', 
             'Thursday', 'Friday', 'Saturday', 'Sunday']

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# %% [markdown]
# Chart 1: Hour of Day

# %%
hourly = df['Created Date'].dt.hour.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(hourly.index, hourly.values, color='steelblue', edgecolor='none')

ax.set_title('Complaint Volume by Hour of Day', fontsize=14, pad=15)
ax.set_xlabel('Hour of Day (0 = Midnight, 12 = Noon)', fontsize=11)
ax.set_ylabel('Number of Complaints', fontsize=11)
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], 
                    rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# %% [markdown]
# Let's go from left to right for this chart. At midnight, there seem to be quite a few complaints at a bit over 125,000 and then quickly drop off. A lot of people are still up at midnight and these are likely noise complaints as everyone is trying to go to bed. It's natural, then, that the graph starts to dip from midnight until around 5AM - 6AM where people have been asleep and are getting up for the day. A quick note here that 4AM seems to be the time of day where the fewest complaints are filed, sitting at just under 50,000.
# 
# As we follow the curve upward from 4AM, we can see that my prediction proved close but slightly off in that complaints are at their highest, entering the 200,000 range at 10AM and hitting its peak at 11AM. This is late morning into early afternoon when most people are going about their daily tasks. From here on out we see things start to slope downward, but not very quickly, staying above the 150,000 total line the entire time.
# 
# Towards the end of the graph, we see that 7PM (19:00) hits a bottom. The number is still high, sitting almost perfectly between 150,000 and 175,000, but begins to climb again until 10PM where it's a bit over 175,000 total complaints. This is likely a spike in noise complaints and illegal parking as people are either home getting ready for bed or getting home from a late shift or night activity. From 10PM we see the shift downwards again, where we come full circle. 
# 

# %% [markdown]
# Chart 2: Day of Week

# %%
daily = df['Created Date'].dt.dayofweek.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(day_names, daily.values, color='steelblue', edgecolor='none')

# Add value labels on top of each bar
for bar, val in zip(bars, daily.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

ax.set_title('Complaint Volume by Day of Week', fontsize=14, pad=15)
ax.set_xlabel('Day of Week', fontsize=11)
ax.set_ylabel('Number of Complaints', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# %% [markdown]
# As we look at the complaints by day of the week, we can immediately see that my prediction was incorrect. The weekends here actually have the lowest complaint total, with Saturday and Sunday (the graph low) being almost identical at just under 472,000 each. However, looking at the weekday section of the graph, it's easy to see that they dominate with Monday leading at roughly 525,000 and a gradual decline to the weekday low of Thursday.
# 
# The reason here is probably more about how the city and its services operate and not so much about who is home. Most of these agencies operate primarily on the weekdays, which explains why we see a higher count. 

# %% [markdown]
# Chart 3: Month

# %%
monthly = df['Created Date'].dt.month.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(month_names, monthly.values, color='steelblue', edgecolor='none')

# Add value labels on top of each bar
for bar, val in zip(bars, monthly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

ax.set_title('Complaint Volume by Month (2025)', fontsize=14, pad=15)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Number of Complaints', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# %% [markdown]
# When we look at the complaint volume by month, January leads the year with just over 340,000 complaints, which isn't surprising given the winter heating issues, frozen pipes, and freezing temperatures create a spike in complaints. February is a sharp drop at the graph low of about 247,000 complaints, though some of that is simply because February is a short month with only 28 days. 
# 
# It's also worth noting that January may have a larger count due to trailing complaints from December 2024 that weren't closed until January 2025. Since the dataset captures complaints by their close date, January becomes the natural spot for complaints opened late 2024 and resolved early in the new year. This same logic also applies on the other end of the spectrum where December 2025 will likely have a lot of year-end complaints unresolved until 2026.
# 
# What's interesting is that we don't see a spike in the summer months like I had predicted. May through September is very consistent without much fluctuation hovering between 280,000 and 300,000 with no clear surge. Where I thought that more people being active in the nicer weather and school being out for the summer would create higher totals, actually wasn't the case.
# 
# While still on the higher side, especially December, the year-end months are worth treating with a bit of caution. Since this dataset only includes records that were fully closed at the time of export, late year complaints may not have been closed yet. As we've seen earlier, some complaints take months to resolve, which may make November and December look smaller. These months probably don't show a real seasonal pattern like the rest of the year does.

# %%
# Split into noise vs non-noise by hour
noise_types = ['Noise - Residential', 'Noise - Street/Sidewalk', 
               'Noise - Commercial', 'Noise - Vehicle', 'Noise']

noise = df[df['Complaint Type'].isin(noise_types)]
non_noise = df[~df['Complaint Type'].isin(noise_types)]

noise_hourly = noise['Created Date'].dt.hour.value_counts().sort_index()
non_noise_hourly = non_noise['Created Date'].dt.hour.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(noise_hourly.index, noise_hourly.values, 
        color='steelblue', label='Noise complaints', linewidth=2)
ax.plot(non_noise_hourly.index, non_noise_hourly.values, 
        color='tomato', label='All other complaints', linewidth=2)

ax.set_title('Complaint Volume by Hour — Noise vs Everything Else', 
             fontsize=14, pad=15)
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Number of Complaints', fontsize=11)
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], 
                    rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# This chart explains that my hypothesis was correct in noise complaints being lower during the day and higher at night vs all other complaints being higher during the day and lower at night. The blue line here indicates any form of "Noise" complaint that we've seen, so that includes residential, street/sidewalk, commercial, and the generic "Noise" complaints. The red line is any other type of complaint in the dataset.
# 
# As you can see, from midnight until between 2AM and 3AM, the lines run fairly parallel with noise complaints being slightly higher. At about 2:15-2:30AM, the lines intersect and the red line starts to climb with a huge spike starting at 5AM as life in NYC starts becoming more active. At that same point, noise complaints drop and sits below the 25,000 mark all morning long. 
# 
# What's interesting is that at 5AM, where we see the spike for all other complaints, we see the noise complaints at their lowest (around 10,000-15,000). This is the shifting point where people are likely waking up and getting ready to go about their day, so complaints will shift from noise to the other complaints we've been seeing. Likewise, we see the noise complaints start to increase again at around 10AM, the same point when all other complaint types start to decrease from their massive peak of over 175,000.
# 
# These 2 lines slowly make their way back to each other where we see noise complaints hit their peak at 10PM (22:00) and then intersect with the red line just before midnight. This clearly shows the correlation between noise complaints being more prominent when people are home vs when they are away.
# 
# This also helps us see the NYPD's dominance in closing complaints quickly since the agency handling most of the noise complaints has officers already active and on patrol at night.

# %% [markdown]
# # Tying it all together
# 
# Over the course of this notebook we've asked and answered 5 different questions, and there's a clear thread connecting all of them. The data 
# isn't random as it tells a real story about how New York City actually operates.
# 
# The fast complaints are almost entirely NYPD-handled, with near-immediate resolution times, while slower complaints follow formal multi-step processes. NYPD handles nearly half of all complaints citywide, and the dominant complaint types are noise and parking, both things officers can resolve on the spot. Brooklyn leads in raw complaint volume, but what people complain about varies by borough. Across all of it, the time patterns show complaints naturally following the rhythms of city life like morning peaks, evening noise, and weekday dominance.
# 
# NYC's 311 data isn't just a boring government record. It's a picture of a city going about its day.

# %% [markdown]
# # Summary
# 
# - The dataset contains 3.6 million records of 2025 and 3.52 million after cleaning to include only valid closed statuses and positive resolution times
# 
# - There were three data quality issues found: DHS had missing close dates (99% DHS), there were negative resolution times (91% DOT/Street Light), and year-plus resolutions (spread across multiple agencies), likely from long-term infrastructure requests and not a data quality issue
# 
# - Resolution times are heavily right-skewed with a median of 7.15 hours vs a mean of 214.7 hours.
# 
# - Resolution time had two distinct peaks: NYPD complaints close in under 2 hours (34% of all complaints) and HPD housing complaints take about a month
# 
# - NYPD handles 49% of all the valid complaints, which is more than the next 4 agencies combined
# 
# - The top individual complaint was Illegal Parking, but Noise complaints are the real volume leader when combining subcategories (about 756,000)
# 
# - Brooklyn reports 29.6% of all complaints, nearly double Staten Island's 3.7%
# 
# - Complaints peak at 11AM and again at 10PM; the evening spike is almost entirely noise complaints
# 
# - Weekdays lead complaint totals despite more people being home on the weekends; city services operate on weekday business hours
# 
# - January contains the highest monthly volume of complaints, partly because of heating issues and partly due to complaints from December 2024 closing in January 2025


