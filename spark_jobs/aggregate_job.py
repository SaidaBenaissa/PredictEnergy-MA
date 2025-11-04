# spark_jobs/clean_job.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

def main(args):
    spark = (
        SparkSession.builder.appName("EnergyClean")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    in_path = args.input
    out_path = args.output

    # Read SILVER (annual)
    df = (
        spark.read.option("header", True).option("inferSchema", True).csv(in_path)
    )

    # Standardize expected columns if needed
    alias_map = {
        "annee": "year", "Année": "year", "Annee": "year",
        "valeur_fossiles": "fossil_pct", "fossiles_pct": "fossil_pct",
        "valeur_petrole": "oil_pct", "petrole_pct": "oil_pct",
        "valeur_renouvhx": "renewables_kWh", "renouvelables": "renewables_kWh",
        "temp": "temperature", "t": "temperature", "temp_moy": "temperature",
    }
    for src, dst in alias_map.items():
        if src in df.columns:
            df = df.withColumnRenamed(src, dst)

    # Keep only needed columns
    cols = ["year","fossil_pct","oil_pct","renewables_kWh","temperature"]
    df = df.select(*[c for c in cols if c in df.columns])

    # Filter years & drop NaN rows
    df_clean = (
        df.where((col("year") >= 1990) & (col("year") <= 2023))
          .dropna()
    )

    # Write cleaned SILVER (overwrite)
    (df_clean.coalesce(1)
        .write.mode("overwrite").option("header", True).csv(out_path))

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="e.g. ../data/silver/maroc_energy_weather_1990_2023.csv")
    p.add_argument("--output", required=True, help="e.g. ../data/silver/silver_clean")
    main(p.parse_args())
