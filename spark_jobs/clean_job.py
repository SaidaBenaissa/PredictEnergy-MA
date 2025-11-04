# spark_jobs/clean_job.py
# Goal: read SILVER (énergie+météo), cast types, and write Parquet partitioned by year.

import os, sys
from pathlib import Path

# --------- Environment hardening (Windows-safe) ----------
# Force PySpark to use this Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Ensure JAVA + PATH + TEMP exist (avoids “cmd introuvable” on some Windows setups)
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-17")
path_fragments = [
    r"C:\Windows\System32",
    r"C:\Windows",
    os.path.join(os.environ["JAVA_HOME"], "bin"),
    os.environ.get("PATH", ""),
]
os.environ["PATH"] = ";".join(p for p in path_fragments if p)

os.environ.setdefault("TEMP", r"C:\Temp")
os.environ.setdefault("TMP", r"C:\Temp")
try:
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
except Exception:
    pass
# --------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def resolve_source_csv(root: Path) -> Path:
    """Return the CSV path; try the known filename, else pick the first csv in data/silver."""
    candidate = root / "data" / "silver" / "maroc_energy_weather_1990_2023.csv"
    if candidate.exists():
        return candidate

    # Fallback: any csv in data/silver
    silver_dir = root / "data" / "silver"
    csvs = sorted(silver_dir.glob("*.csv"))
    if csvs:
        return csvs[0]

    raise FileNotFoundError(
        f"CSV introuvable. Cherché: {candidate}\n"
        f"Dossier listé: {silver_dir}\n"
        f"Contenu: {[p.name for p in silver_dir.glob('*')]}"
    )

def main():
    ROOT = Path(__file__).resolve().parents[1]
    src  = resolve_source_csv(ROOT)
    out  = ROOT / "data" / "silver_spark" / "energy_weather_parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("CleanSilver")
        .config("spark.sql.session.timeZone", "UTC")
        # safer local mode on Windows
        .config("spark.master", "local[2]")
        # parquet/arrow niceties
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

    print(f"[INFO] Using Python: {sys.executable}")
    print(f"[INFO] JAVA_HOME  : {os.environ.get('JAVA_HOME')}")
    print(f"[INFO] Source CSV : {src}")
    print(f"[INFO] Output     : {out}")

    # Read
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(src))
    )

    # Handle "annee" or "année"
    year_col = "annee" if "annee" in df.columns else ("année" if "année" in df.columns else None)
    if not year_col:
        raise KeyError(f"Aucune colonne 'annee' ou 'année' trouvée. Colonnes: {df.columns}")

    # Casts
    def safe_cast(dframe):
        cols = {
            year_col: "int",
            "valeur_fossiles": "double",
            "valeur_petrole": "double",
            "valeur_renouvhx": "double",
            "temperature": "double",
        }
        for c, t in cols.items():
            if c in dframe.columns:
                dframe = dframe.withColumn(c, col(c).cast(t))
        return dframe

    df = safe_cast(df).withColumnRenamed(year_col, "year")

    # Write Parquet partitioned by year
    (
        df.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(str(out))
    )

    print(f"[OK] Wrote Parquet to: {out}")
    spark.stop()

if __name__ == "__main__":
    main()
