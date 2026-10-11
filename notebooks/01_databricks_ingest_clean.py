# Databricks notebook source
# TravelIQ - Day 1
# Load raw hotel booking data using PySpark

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/workspace/default/travel_iq_data/hotel_bookings.csv")
)

display(df)

# COMMAND ----------

# Check the structure of the dataset

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns)

print("\nData types:")
df.printSchema()

# COMMAND ----------

# Check missing values in every column

from pyspark.sql.functions import col, sum

null_counts = df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

display(null_counts)

# COMMAND ----------

# Show only columns that contain NULL values

from pyspark.sql.functions import col, sum

null_counts = df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

# Convert the result into column-value pairs
null_summary = null_counts.selectExpr(
    "stack(32, " +
    ", ".join([f"'{c}', `{c}`" for c in df.columns]) +
    ") as (column_name, null_count)"
)

# Keep only columns with NULL values
display(
    null_summary
    .filter(col("null_count") > 0)
    .orderBy(col("null_count").desc())
)

# COMMAND ----------

# Check for empty strings and NULL-like values

from pyspark.sql.functions import col, sum, when, trim, lower

missing_like = df.select([
    sum(
        when(
            col(c).isNull() |
            (trim(col(c).cast("string")) == "") |
            (lower(trim(col(c).cast("string"))).isin("null", "na", "n/a")),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df.columns
])

display(missing_like)

# COMMAND ----------

# Show only columns having missing-like values

from pyspark.sql.functions import col, sum, when, trim, lower

missing_like = df.select([
    sum(
        when(
            col(c).isNull() |
            (trim(col(c).cast("string")) == "") |
            (lower(trim(col(c).cast("string"))).isin("null", "na", "n/a")),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df.columns
])

# Convert result to a Python dictionary
missing_values = missing_like.collect()[0].asDict()

print("Columns with missing-like values:\n")

for column, count in missing_values.items():
    if count > 0:
        print(f"{column}: {count}")

# COMMAND ----------

# Check for duplicate records

total_rows = df.count()
unique_rows = df.dropDuplicates().count()
duplicate_rows = total_rows - unique_rows

print("Total rows:", total_rows)
print("Unique rows:", unique_rows)
print("Duplicate rows:", duplicate_rows)

# COMMAND ----------

# Find groups of completely identical records

duplicate_groups = (
    df.groupBy(df.columns)
      .count()
      .filter(col("count") > 1)
)

print("Number of duplicate groups:", duplicate_groups.count())

display(duplicate_groups.orderBy(col("count").desc()).limit(10))

# COMMAND ----------

# Remove exact duplicate records

df_clean = df.dropDuplicates()

print("Original rows:", df.count())
print("Rows after removing exact duplicates:", df_clean.count())
print("Rows removed:", df.count() - df_clean.count())

# COMMAND ----------

# Check missing-like values after duplicate removal

from pyspark.sql.functions import col, sum, when, trim, lower

missing_clean = df_clean.select([
    sum(
        when(
            col(c).isNull() |
            (trim(col(c).cast("string")) == "") |
            (lower(trim(col(c).cast("string"))).isin("null", "na", "n/a")),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df_clean.columns
])

missing_values_clean = missing_clean.collect()[0].asDict()

print("Missing-like values after duplicate removal:\n")

for column, count in missing_values_clean.items():
    if count > 0:
        print(f"{column}: {count}")

# COMMAND ----------

# Check agent and company value distributions

print("Agent distribution:")
display(
    df_clean.groupBy("agent")
    .count()
    .orderBy(col("count").desc())
    .limit(15)
)

print("Company distribution:")
display(
    df_clean.groupBy("company")
    .count()
    .orderBy(col("count").desc())
    .limit(15)
)

# COMMAND ----------

# Business-aware missing value treatment

from pyspark.sql.functions import col, when, trim

df_final = (
    df_clean

    # Children: keep valid numeric values, otherwise 0
    .withColumn(
        "children",
        when(
            trim(col("children").cast("string")).rlike("^[0-9]+$"),
            col("children").cast("int")
        ).otherwise(0)
    )

    # Country: missing/invalid text -> Unknown
    .withColumn(
        "country",
        when(
            col("country").isNull() |
            (trim(col("country").cast("string")) == "") |
            (trim(col("country").cast("string")).isin("NULL", "NA", "N/A")),
            "Unknown"
        )
        .otherwise(col("country"))
    )

    # Agent: only numeric values are valid, everything else -> 0
    .withColumn(
        "agent",
        when(
            trim(col("agent").cast("string")).rlike("^[0-9]+$"),
            col("agent").cast("long")
        ).otherwise(0)
    )

    # Company: only numeric values are valid, everything else -> 0
    .withColumn(
        "company",
        when(
            trim(col("company").cast("string")).rlike("^[0-9]+$"),
            col("company").cast("long")
        ).otherwise(0)
    )
)

print("Rows in final cleaned dataset:", df_final.count())

# COMMAND ----------

# Check final data types after cleaning

df_final.printSchema()

# COMMAND ----------

# Final validation after cleaning

print("Final row count:", df_final.count())
print("Final column count:", len(df_final.columns))

print("\nRemaining NULL values:")

from pyspark.sql.functions import col, sum

final_nulls = df_final.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df_final.columns
])

display(final_nulls)

# COMMAND ----------

# Save final cleaned dataset as Parquet

output_path = "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"

df_final.write \
    .mode("overwrite") \
    .parquet(output_path)

print("Cleaned dataset saved successfully!")
print("Path:", output_path)

# COMMAND ----------

# Verify saved cleaned dataset

df_check = spark.read.parquet(
    "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"
)

print("Rows:", df_check.count())
print("Columns:", len(df_check.columns))

display(df_check.limit(10))

# COMMAND ----------

# TravelIQ - Day 2
# Load cleaned dataset for analysis

df = spark.read.parquet(
    "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"
)

print("Rows:", df.count())
print("Columns:", len(df.columns))

display(df.limit(10))

# COMMAND ----------

# Overall Cancellation Rate

from pyspark.sql.functions import col, avg

cancellation_rate = (
    df.select(
        (avg(col("is_canceled")) * 100).alias("cancellation_rate")
    )
)

display(cancellation_rate)

# COMMAND ----------

#  Cancellation Rate by Hotel

from pyspark.sql.functions import col, avg, round

hotel_cancellation = (
    df.groupBy("hotel")
      .agg(
          round(avg(col("is_canceled")) * 100, 2)
          .alias("cancellation_rate")
      )
      .orderBy(col("cancellation_rate").desc())
)

display(hotel_cancellation)

# COMMAND ----------

# Cancellation Rate by Year

yearly_cancellation = (
    df.groupBy("arrival_date_year")
      .agg(
          round(avg(col("is_canceled")) * 100, 2)
          .alias("cancellation_rate")
      )
      .orderBy("arrival_date_year")
)

display(yearly_cancellation)

# COMMAND ----------

# Step 4 - Cancellation Rate by Lead Time

from pyspark.sql.functions import when

lead_time_analysis = (
    df.withColumn(
        "lead_time_group",
        when(col("lead_time") < 30, "0-29 days")
        .when(col("lead_time") < 90, "30-89 days")
        .when(col("lead_time") < 180, "90-179 days")
        .otherwise("180+ days")
    )
    .groupBy("lead_time_group")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
)

display(lead_time_analysis)

# COMMAND ----------

# Cancellation rate by market segment
from pyspark.sql.functions import col, avg, round
market_segment_analysis = (
    df
    .groupBy("market_segment")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(market_segment_analysis)

# COMMAND ----------

# Cancellation rate by booking changes

booking_changes_analysis = (
    df
    .groupBy("booking_changes")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("booking_changes"))
)

display(booking_changes_analysis)

# COMMAND ----------



# Cancellation rate by deposit type

deposit_analysis = (
    df
    .groupBy("deposit_type")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(deposit_analysis)

# COMMAND ----------

# Load the saved cleaned dataset

df_final = spark.read.parquet(
    "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"
)

print("Rows:", df_final.count())
print("Columns:", len(df_final.columns))

# COMMAND ----------

# Cancellation rate by deposit type

from pyspark.sql.functions import col, avg, round

deposit_analysis = (
    df_final
    .groupBy("deposit_type")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(deposit_analysis)

# COMMAND ----------

# Load cleaned TravelIQ dataset

df_final = spark.read.parquet(
    "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"
)

print("Rows:", df_final.count())
print("Columns:", len(df_final.columns))

# COMMAND ----------

# Cancellation rate by customer type

from pyspark.sql.functions import col, avg, round

customer_type_analysis = (
    df_final
    .groupBy("customer_type")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(customer_type_analysis)

# COMMAND ----------

# Cancellation rate by meal type

meal_analysis = (
    df_final
    .groupBy("meal")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(meal_analysis)

# COMMAND ----------

# Cancellation rate by reserved room type

room_analysis = (
    df_final
    .groupBy("reserved_room_type")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(room_analysis)

# COMMAND ----------

# Cancellation rate by previous cancellations

from pyspark.sql.functions import col, avg, round

previous_cancellation_analysis = (
    df_final
    .groupBy("previous_cancellations")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("previous_cancellations"))
)

display(previous_cancellation_analysis)

# COMMAND ----------

# Cancellation rate by booking changes

booking_changes_analysis = (
    df_final
    .groupBy("booking_changes")
    .agg(
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("booking_changes"))
)

display(booking_changes_analysis)

# COMMAND ----------

# Overall cancellation summary by hotel and customer type
from pyspark.sql.functions import col, count, round

hotel_customer_analysis = (
    df_final
    .groupBy("hotel", "customer_type")
    .agg(
        count("*").alias("total_bookings"),
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(col("cancellation_rate").desc())
)

display(hotel_customer_analysis)

# COMMAND ----------

# Cancellation impact by hotel
from pyspark.sql.functions import col, avg, round, count, sum
cancellation_impact = (
    df_final
    .groupBy("hotel")
    .agg(
        count("*").alias("total_bookings"),
        sum(col("is_canceled")).alias("cancelled_bookings"),
        round(avg(col("is_canceled")) * 100, 2).alias("cancellation_rate")
    )
    .orderBy(col("cancelled_bookings").desc())
)

display(cancellation_impact)

# COMMAND ----------

# Create lead-time based cancellation risk category
from pyspark.sql.functions import col, avg, round, count, sum, when
df_risk = (
    df_final
    .withColumn(
        "lead_time_risk",
        when(col("lead_time") < 30, "Low")
        .when(col("lead_time") < 90, "Medium")
        .when(col("lead_time") < 180, "High")
        .otherwise("Very High")
    )
)

display(
    df_risk
    .groupBy("lead_time_risk")
    .agg(
        count("*").alias("total_bookings"),
        sum(col("is_canceled")).alias("cancelled_bookings"),
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy("cancellation_rate", ascending=False)
)

# COMMAND ----------

# Combined cancellation risk by hotel and lead-time category
from pyspark.sql.functions import col, avg, round, count, sum
combined_risk = (
    df_risk
    .groupBy("hotel", "lead_time_risk")
    .agg(
        count("*").alias("total_bookings"),
        sum(col("is_canceled")).alias("cancelled_bookings"),
        round(avg(col("is_canceled")) * 100, 2)
        .alias("cancellation_rate")
    )
    .orderBy(
        col("hotel"),
        col("cancellation_rate").desc()
    )
)

display(combined_risk)

# COMMAND ----------

df_check = spark.read.parquet(
    "/Volumes/workspace/default/travel_iq_data/hotel_bookings_clean"
)

print(df_check.columns)

# COMMAND ----------

df_check.printSchema()

# COMMAND ----------

df_final.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("/Volumes/workspace/default/travel_iq_data/hotel_bookings_csv")

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/travel_iq_data/hotel_bookings_csv"))