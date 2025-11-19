import streamlit as st
import pandas as pd
import plotly.express as px 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# For forecasting
from prophet import Prophet
from prophet.plot import plot_plotly

# For Spark + Parquet
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

# ================================================
# 🎯 STREAMLIT TITLE
# ================================================
st.title("⚡ Big Data Energy Consumption Dashboard: Morocco")

# =================================================
# 📌 PREDICTIONS SECTION
# =================================================
st.header("📂 Predictions Comparison")

csv_file = "gold_with_predictions.csv"
df_pred = pd.read_csv(csv_file)

st.write("### 📄 Data Preview")
st.dataframe(df_pred)

# ---------------- Plot Real vs Predicted ----------------
fig_pred = px.line(
    df_pred,
    x="date",
    y=["Real_Value", "Pred_LSTM", "Pred_Prophet"],
    title="🔮 Real vs Predicted Consumption"
)
st.plotly_chart(fig_pred)

# =================================================
# 📊 EVALUATION METRICS
# =================================================
st.header("📏 Model Evaluation Metrics")

y_true = df_pred["Real_Value"]
y_lstm = df_pred["Pred_LSTM"]
y_prophet = df_pred["Pred_Prophet"]

# RMSE, MAE, MAPE
rmse_lstm = np.sqrt(mean_squared_error(y_true, y_lstm))
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_prophet))
mae_lstm = mean_absolute_error(y_true, y_lstm)
mae_prophet = mean_absolute_error(y_true, y_prophet)
mape_lstm = np.mean(np.abs((y_true - y_lstm) / y_true)) * 100
mape_prophet = np.mean(np.abs((y_true - y_prophet) / y_true)) * 100

# Display metrics
st.metric("🔷 LSTM RMSE", f"{rmse_lstm:.2f}")
st.metric("🔶 Prophet RMSE", f"{rmse_prophet:.2f}")

st.write("### 📌 MAE & MAPE")
st.write(f"🔷 LSTM MAE: **{mae_lstm:.2f}**, MAPE: **{mape_lstm:.2f}%**")
st.write(f"🔶 Prophet MAE: **{mae_prophet:.2f}**, MAPE: **{mape_prophet:.2f}%**")

# ---------------- Best Model Detection ----------------
best_model = "LSTM" if rmse_lstm < rmse_prophet else "Prophet"
best_rmse = min(rmse_lstm, rmse_prophet)

st.subheader("🥇 Best Performing Model")
st.success(f"Best model is **{best_model}** with RMSE = **{best_rmse:.2f}**")

# =================================================
# 🔎 RESIDUAL ERROR ANALYSIS
# =================================================
st.header("🔎 Residual Error Analysis")

df_pred["residual_lstm"] = y_true - y_lstm
df_pred["residual_prophet"] = y_true - y_prophet

fig_res_line = px.line(
    df_pred,
    x="date",
    y=["residual_lstm", "residual_prophet"],
    title="📉 Residual Errors Over Time"
)
st.plotly_chart(fig_res_line)

fig_res_hist = px.histogram(
    df_pred,
    x=["residual_lstm", "residual_prophet"],
    barmode="overlay",
    nbins=40,
    title="📊 Residual Error Distribution"
)
st.plotly_chart(fig_res_hist)

# =================================================
# 🔮 FUTURE FORECAST WITH PROPHET
# =================================================
st.header("🔮 Future Energy Consumption Forecast")

df_prophet = df_pred[["date", "Real_Value"]].rename(columns={"date":"ds", "Real_Value":"y"})
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_model.fit(df_prophet)

future_periods = st.slider("Select number of months to forecast:", 1, 36, 12)
future = prophet_model.make_future_dataframe(periods=future_periods, freq='M')
forecast = prophet_model.predict(future)

fig_forecast = plot_plotly(prophet_model, forecast)
st.plotly_chart(fig_forecast)

st.subheader("📈 Forecasted Values")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_periods))

# =================================================
# 🌍 PARQUET VISUALIZATION WITH SPARK
# =================================================
st.header("🌍 Energy & Weather Analytics (Parquet + Spark)")

# Spark session
spark = SparkSession.builder.appName("Energy & Weather").getOrCreate()

parquet_path = "file:///home/lenovo/PredictEnergy-MA/data/silver_spark/silver_spark_energy_weather_parquet"
df_pq = spark.read.option("recursiveFileLookup", "true").parquet(parquet_path)

st.write("### 🔎 Sample Data (Parquet)")
st.dataframe(df_pq.limit(10).toPandas())

# ---------------- Average consumption per year ----------------
st.subheader("📊 Average Energy Consumption by Type")

avg_energy = df_pq.groupBy("annee").agg(
    mean("valeur_fossiles").alias("Fossiles"),
    mean("valeur_petrole").alias("Petrole"),
    mean("valeur_renouvhx").alias("Renewables")
).toPandas().sort_values("annee")

fig_avg = px.line(avg_energy, x="annee", y=["Fossiles","Petrole","Renewables"],
                  title="🔆 Average Energy Consumption per Year")
st.plotly_chart(fig_avg)

# ---------------- Temperature Trend ----------------
st.subheader("🌡️ Temperature Trend in Morocco")

temp_pd = df_pq.select("annee", "temperature").toPandas().sort_values("annee")
fig_temp = px.line(temp_pd, x="annee", y="temperature",
                   title="🔥 Annual Average Temperature in Morocco")
st.plotly_chart(fig_temp)

st.success("🎉 All visualizations successfully loaded!")

# Stop Spark
spark.stop()
