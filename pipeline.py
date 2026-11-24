"""Distributed analysis and classification of US traffic accident records."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, functions as F, types as T


# Supply data types without performing a separate schema-inference scan.
# CSV schemas are positional: preserve the exact source header order.
TYPES = {
    "Severity": T.IntegerType(),
    "Temperature(F)": T.DoubleType(),
    "Humidity(%)": T.DoubleType(),
    "Visibility(mi)": T.DoubleType(),
    "Junction": T.BooleanType(),
    "Traffic_Signal": T.BooleanType(),
    "State": T.StringType(),
    "Weather_Condition": T.StringType(),
    "Start_Time": T.StringType(),
}


def source_schema(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        names = next(csv.reader(file))
    missing = set(TYPES) - set(names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return T.StructType([
        T.StructField(name, TYPES.get(name, T.StringType()), True)
        for name in names
    ])


def run(spark, path, output):
    sc = spark.sparkContext
    print(f"Source size: {path.stat().st_size / 1e9:.2f} GB")
    raw = (spark.read.option("header", True).option("escape", '"')
           .schema(source_schema(path)).csv(str(path)))

    # Select required columns and apply native predicates before feature engineering.
    clean = (
        raw.select(*TYPES)
        .filter(F.col("Severity").between(1, 4))
        .filter(F.col("Visibility(mi)").between(0, 140))
        .filter(F.col("Humidity(%)").between(0, 100))
        .filter(F.col("Temperature(F)").between(-100, 150))
        .dropna(subset=["State", "Weather_Condition", "Start_Time", "Junction", "Traffic_Signal"])
        .withColumn("accident_hour", F.hour(F.try_to_timestamp(
            F.substring("Start_Time", 1, 19), F.lit("yyyy-MM-dd HH:mm:ss")
        )))
        .filter(F.col("accident_hour").isNotNull())
        .withColumn("high_severity", (F.col("Severity") >= 3).cast("double"))
        .select(
            F.col("State").alias("state"),
            F.col("Weather_Condition").alias("weather"),
            F.col("Severity").alias("severity"),
            F.col("Temperature(F)").alias("temperature"),
            F.col("Humidity(%)").alias("humidity"),
            F.col("Visibility(mi)").alias("visibility"),
            "accident_hour",
            "high_severity",
            F.col("Junction").cast("double").alias("junction"),
            F.col("Traffic_Signal").cast("double").alias("traffic_signal"),
        )
    )

    # Demonstrate lazy transformation construction versus an action.
    start = perf_counter()
    subset = clean.filter(F.col("high_severity") == 1)
    print(f"\nLazy filter construction: {perf_counter() - start:.4f} seconds")
    sc.setJobDescription("Lazy evaluation demonstration: count higher-severity records")
    start = perf_counter()
    higher_count = subset.count()
    print(f"Action count(): {higher_count:,} rows in {perf_counter() - start:.2f} seconds")

    # A complex aggregation replaces the need for a join. The sum of the binary
    # label equals the number of higher-severity records in each group.
    def aggregate(df):
        return df.groupBy("state", "weather").agg(
            F.count("*").alias("accident_count"),
            F.sum("high_severity").cast("long").alias("high_severity_count"),
            F.avg("severity").alias("mean_severity"),
            F.avg("visibility").alias("mean_visibility"),
        )

    summary = aggregate(clean)
    print("\n=== Physical execution plan (before caching) ===")
    summary.explain(mode="formatted")
    sc.setJobDescription("Uncached state-weather aggregation")
    start = perf_counter()
    uncached_rows = summary.collect()  # only the compact grouped result reaches the driver
    uncached_time = perf_counter() - start
    print(f"Uncached aggregation: {uncached_time:.2f} seconds")
    print("\n=== Executed aggregation plan ===")
    summary.explain(mode="formatted")

    # Cache only the projected, cleaned data. Materialize it before measuring reuse.
    clean.cache()
    sc.setJobDescription("Materialize cleaned data cache")
    start = perf_counter()
    cleaned_count = clean.count()
    print(f"Cache materialization: {perf_counter() - start:.2f} seconds")
    cached_summary = aggregate(clean)
    sc.setJobDescription("Cached state-weather aggregation")
    start = perf_counter()
    cached_rows = cached_summary.collect()
    cached_time = perf_counter() - start
    print(f"Cached aggregation: {cached_time:.2f} seconds")
    print(f"Cleaned records: {cleaned_count:,}; higher severity: {higher_count:,}")
    print(f"Aggregation groups: {len(uncached_rows):,} uncached / {len(cached_rows):,} cached")
    print(f"Cleaned higher-severity share: {higher_count / cleaned_count:.2%}")
    print("\n=== Cached execution plan ===")
    cached_summary.explain(mode="formatted")

    # Reuse the compact aggregated table for both SQL queries.
    # No extra raw-data shuffle or global ordering is needed before grouping.
    grouped = spark.createDataFrame(cached_rows, cached_summary.schema)
    grouped.createOrReplaceTempView("accident_summary")
    state_results = spark.sql("""
        SELECT state, SUM(accident_count) AS accident_count,
               ROUND(100.0 * SUM(high_severity_count) / SUM(accident_count), 2)
                   AS high_severity_pct
        FROM accident_summary
        GROUP BY state
        HAVING SUM(accident_count) >= 1000
        ORDER BY high_severity_pct DESC, state
    """)
    weather_results = spark.sql("""
        SELECT weather, SUM(accident_count) AS accident_count,
               ROUND(SUM(mean_severity * accident_count) / SUM(accident_count), 3)
                   AS mean_severity,
               ROUND(100.0 * SUM(high_severity_count) / SUM(accident_count), 2)
                   AS high_severity_pct
        FROM accident_summary
        GROUP BY weather
        HAVING SUM(accident_count) >= 1000
        ORDER BY high_severity_pct DESC, weather
    """)
    sc.setJobDescription("SQL query 1: higher-severity share by state")
    print("\n=== SQL 1: States with the highest higher-severity shares ===")
    state_results.show(10, truncate=False)
    sc.setJobDescription("SQL query 2: severity by weather")
    print("\n=== SQL 2: Weather conditions with the highest higher-severity shares ===")
    weather_results.show(10, truncate=False)

    # The outputs are small, so a single Parquet part per table avoids tiny files.
    for name, table in [("state_weather", grouped), ("by_state", state_results),
                        ("by_weather", weather_results)]:
        sc.setJobDescription(f"Write {name} Parquet")
        table.coalesce(1).write.mode("overwrite").parquet(str(output / name))

    # Simple MLlib classification. The analysis above uses all cleaned records;
    # only the ML step is sampled to keep laptop runtime manageable.
    features = ["temperature", "humidity", "visibility", "accident_hour",
                "junction", "traffic_signal"]
    sample = clean.select(*features, "high_severity").sample(False, 0.05, seed=42)
    model_data = VectorAssembler(inputCols=features, outputCol="features").transform(sample)
    train, test = model_data.select("features", "high_severity").randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    sc.setJobDescription("MLlib logistic regression training")
    model = LogisticRegression(labelCol="high_severity", maxIter=20, regParam=0.01).fit(train)
    predictions = model.transform(test).cache()
    print("\n=== MLlib logistic regression ===")
    print(f"Training records: {train.count():,}; test records: {test.count():,}")
    counts = predictions.groupBy("high_severity").count().collect()
    baseline = max(row["count"] for row in counts) / sum(row["count"] for row in counts)
    print(f"Majority-class accuracy baseline: {baseline:.4f}")
    for metric in ["accuracy", "f1"]:
        value = MulticlassClassificationEvaluator(
            labelCol="high_severity", metricName=metric
        ).evaluate(predictions)
        print(f"{metric} (weighted for F1): {value:.4f}")
    auc = BinaryClassificationEvaluator(labelCol="high_severity", metricName="areaUnderROC")
    print(f"ROC-AUC: {auc.evaluate(predictions):.4f}")
    predictions.unpersist()
    train.unpersist()
    test.unpersist()
    clean.unpersist()
    print(f"\nSUCCESS: Pipeline completed. Parquet results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/US_Accidents_March23.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--hold-ui", action="store_true", help="Keep Spark UI open for screenshots")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"CSV not found: {args.input}")
    spark = (
        SparkSession.builder.appName("US Traffic Accident Analysis")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.csv.filterPushdown.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version} | UI: {spark.sparkContext.uiWebUrl}")
    try:
        run(spark, args.input.resolve(), args.output)
        if args.hold_ui:
            input("\nCapture Spark UI screenshots, then press Enter to stop Spark. ")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
