import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Parquet file
df = pd.read_parquet("flights_RUH.parquet")


# SECTION 1: Initial Data Overview

# Data Overview
print("First 5 rows:\n", df.head())
print("\nData shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns)

# General information & data types
print("\nColumn info, data types, and non-null counts:\n")
print(df.info())
print("\nStatistical summary of numeric columns:\n")
print(df.describe())

# Number of unique values per column with handling complex columns
print("\nNumber of unique values per column:")
for col in df.columns:
    try:
        print(f"{col}: {df[col].nunique()}")
    except TypeError:
        print(f"{col}: contains non-countable values -> converting to string")
        df[col] = df[col].astype(str)
        print(f"{col} (after conversion): {df[col].nunique()}")
# Q1: What are the basic statistics of the flights in the dataset, including total number of flights, unique airline companies, and different aircraft types used?
total_flights = len(df)
print("Total flights:", total_flights)

num_airlines = df['airline.name'].unique()
print("Total airlines copmanies:",num_airlines)

num_aircraft_type = df['aircraft.model'].nunique()
print("Total of aircraft types:",num_aircraft_type)

# SECTION 2: Airline Analysis

# Q2: Which airlines have the highest number of flights, and what are the top 10 airlines by flight count?
flights_per_airline = df['airline.name'].value_counts()
print(flights_per_airline)
# Visualization
flights_per_airline.head(10).plot(kind='bar', figsize=(10,6),
title='Top 10 Airlines by Number of Flights')
plt.xlabel('Airline Name')
plt.ylabel('Number of Flights')
plt.show()

# Q3: How many flights of each status does each airline have?
status_by_airline = df.groupby(['airline.name','status']).size().unstack(fill_value=0)
print(status_by_airline)

# Q4: Which airlines operate the highest average number of flights per day,
# and what are the top 10 airlines by average daily flights?
df['departure_date'] = pd.to_datetime(df['movement.scheduledTime.utc']).dt.date
daily_counts = df.groupby(['airline.name', 'departure_date']).size().reset_index(name='daily_flights')
avg_daily_flights = daily_counts.groupby('airline.name')['daily_flights'] \
                                .mean() \
                                .round(2) \
                                .sort_values(ascending=False) \
                                .reset_index(name='avg_daily_flights')
print(avg_daily_flights.head(10))

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(x='avg_daily_flights', y='airline.name',
            data=avg_daily_flights.head(10), palette='viridis')
plt.title('Top 10 Airlines by Average Daily Flights')
plt.xlabel('Average Daily Flights')
plt.ylabel('Airline')
plt.show()

# SECTION 3: Aircraft Analysis

# Q5: What are the top 10 most frequently used aircraft models ?
aircraft_model_counts = df['aircraft.model'].value_counts()
print(aircraft_model_counts.head(10))


# SECTION 4: Time-Based Analysis

# Q6: Which hours of the day have the highest number of flights, and what are the top 10 busiest hours?
df['movement.scheduledTime.local'] = pd.to_datetime(df['movement.scheduledTime.local'], errors='coerce')
df['hour'] = df['movement.scheduledTime.local'].dt.hour
hourly_distribution = df['hour'].value_counts()
top_10_hours = hourly_distribution.nlargest(10)
print(top_10_hours)
# Visualization
top_10_hours.plot(kind='bar', figsize=(12,5), title='Top 10 hours by number of flight')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Flights')
plt.show()

# Q7: What is the average number of flights per day?
df['movement.scheduledTime.local'] = pd.to_datetime(df['movement.scheduledTime.local'], errors='coerce')
df['date'] = df['movement.scheduledTime.local'].dt.date
daily_avg = df.groupby('date').size().mean()
print("Average flights per day:", round(daily_avg))

# Q8: How do flight statuses vary across different hours of the day?
df['movement.scheduledTime.local'] = pd.to_datetime(
    df['movement.scheduledTime.local'], errors='coerce'
)
df = df.dropna(subset=['movement.scheduledTime.local'])

df['hour'] = df['movement.scheduledTime.local'].dt.hour

status_by_hour = df.groupby(['hour', 'status']).size().unstack(fill_value=0)
print(status_by_hour)

# Q9: What are the top 5 days with the highest number of flights?
df['departure_date'] = pd.to_datetime(df['movement.scheduledTime.local']).dt.date

daily_flights = df.groupby('departure_date').size().reset_index(name='num_flights')
top5_days = daily_flights.sort_values(by='num_flights', ascending=False).head(5)
print(top5_days)

# SECTION 5: Airport Analysis

# Q10: Which airports have the highest number of departing flights?
origin_counts = df['origin_airport_name'].value_counts()
print(origin_counts)

# Q11: Which airports receive the highest number of flights?
dest_counts = df['destination_airport_name'].value_counts()
print(dest_counts.head(10))

# Q12: Which destination airports are served by the highest number
# of airlines, and what are the top 10 busiest airports by airline diversity?
airport_airlines = df.groupby("destination_airport_name")['airline.name'] \
                     .nunique() \
                     .reset_index(name='num_airlines') \
                     .sort_values(by='num_airlines', ascending=False)
print(airport_airlines.head(10))

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(x='num_airlines', y='destination_airport_name',
            data=airport_airlines.head(10), palette='viridis')
plt.title('Top 10 Destination Airports by Number of Airlines')
plt.xlabel('Number of Airlines')
plt.ylabel('Destination Airport')
plt.show()

# Q13: How are flights distributed across different airport time zones?
timezone_counts = df['movement.airport.timeZone'].value_counts()
print(timezone_counts)


# َََQ14: Which airline–origin airport combinations have the most flights?
flights_airline_origin = df.groupby(['airline.name','origin_airport_name']).size()
print(flights_airline_origin.head(10))

# Q15: Which destination airports receive the highest number of flights, and what are the top 10 destination airports by flight volume?
top10_dest = df.groupby('destination_airport_name').size().reset_index(name='num_flights')
top10_dest = top10_dest.sort_values(by='num_flights', ascending=False).head(10)
print(top10_dest)

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(x='num_flights', y='destination_airport_name', data=top10_dest, palette='viridis')
plt.title('Top 10 Destination Airports by Number of Flights')
plt.xlabel('Number of Flights')
plt.ylabel('Destination Airport')
plt.show()




