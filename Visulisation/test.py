import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 🔮 Prévisions
from prophet import Prophet
from prophet.plot import plot_plotly

# 💡 Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord Énergie - Maroc",
    page_icon="⚡",
    layout="wide"
)

# 🎨 Style CSS personnalisé
st.markdown("""
<style>
/* Title style */
h1 {
    text-align: center;
    color: #2a4d69;
    font-weight: 900;
}
/* Section headers */
.section-title {
    background-color: #2a4d69;
    color: white;
    padding: 10px;
    border-radius: 8px;
    font-size:22px;
    font-weight:700;
}
/* Success box */
.stSuccess {
    background-color: #e9f7ef !important;
}
/* Dataframe container */
.dataframe {
    border-radius: 10px;
}
/* Slider style */
.css-1v0mbdj {
    color: #2a4d69 !important;
}
</style>
""", unsafe_allow_html=True)

# ================================================
# 🎯 TITRE & INTRO
# ================================================
st.title(" **Tableau de Bord de la Consommation d’Énergie au Maroc**")
st.markdown("<h5 style='text-align:center; color:#556;'>Analyse de données réelles + Prédiction par modèles + Prévisions futures</h5>", unsafe_allow_html=True)
st.write("---")

# =================================================
# 📌 CHARGEMENT DES DONNÉES
# =================================================
st.markdown("<div class='section-title'> Consommation réelle vs prédite</div>", unsafe_allow_html=True)

csv_file = "gold_with_predictions.csv"
df_pred = pd.read_csv(csv_file)

with st.expander(" Afficher les données brutes"):
    st.dataframe(df_pred, use_container_width=True)

fig_pred = px.line(
    df_pred,
    x="date",
    y=["Real_Value", "Pred_LSTM", "Pred_Prophet"],
    markers=True,
    title="Consommation d’électricité : Réelle vs Prédite",
    color_discrete_sequence=["#2a4d69", "#4b86b4", "#adcbe3"],
    template="plotly_white"
)
st.plotly_chart(fig_pred, use_container_width=True)
st.write("---")

# =================================================
# 📊 ÉVALUATION DES MODÈLES
# =================================================
st.markdown("<div class='section-title'> Évaluation des performances des modèles</div>", unsafe_allow_html=True)

y_true = df_pred["Real_Value"]
y_lstm = df_pred["Pred_LSTM"]
y_prophet = df_pred["Pred_Prophet"]

metrics = {
    "LSTM": {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_lstm)),
        "MAE": mean_absolute_error(y_true, y_lstm),
        "MAPE": np.mean(np.abs((y_true - y_lstm) / y_true)) * 100
    },
    "Prophet": {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_prophet)),
        "MAE": mean_absolute_error(y_true, y_prophet),
        "MAPE": np.mean(np.abs((y_true - y_prophet) / y_true)) * 100
    }
}

best_model = min(metrics, key=lambda m: metrics[m]["RMSE"])
st.success(f" **Meilleur modèle de prédiction : `{best_model}`** (Basé sur le RMSE le plus faible)")

col1, col2, col3 = st.columns(3)
col1.metric("🔷 RMSE LSTM", f"{metrics['LSTM']['RMSE']:.2f}")
col1.metric("🔶 RMSE Prophet", f"{metrics['Prophet']['RMSE']:.2f}")
col2.metric("🔷 MAE LSTM", f"{metrics['LSTM']['MAE']:.2f}")
col2.metric("🔶 MAE Prophet", f"{metrics['Prophet']['MAE']:.2f}")
col3.metric("🔷 MAPE LSTM (%)", f"{metrics['LSTM']['MAPE']:.2f}%")
col3.metric("🔶 MAPE Prophet (%)", f"{metrics['Prophet']['MAPE']:.2f}%")
st.write("---")

# =================================================
# 🔎 ANALYSE DES ERREURS RÉSIDUELLES
# =================================================
st.markdown("<div class='section-title'> Analyse des erreurs résiduelles</div>", unsafe_allow_html=True)

df_pred["residual_lstm"] = y_true - y_lstm
df_pred["residual_prophet"] = y_true - y_prophet

col_a, col_b = st.columns(2)

with col_a:
    fig_res_line = px.line(
        df_pred,
        x="date",
        y=["residual_lstm", "residual_prophet"],
        title="Évolution des erreurs résiduelles",
        template="plotly_white"
    )
    st.plotly_chart(fig_res_line, use_container_width=True)

with col_b:
    fig_res_hist = px.histogram(
        df_pred,
        x=["residual_lstm", "residual_prophet"],
        barmode="overlay",
        nbins=40,
        title="Distribution des erreurs résiduelles",
        template="plotly_white"
    )
    st.plotly_chart(fig_res_hist, use_container_width=True)
st.write("---")

# =================================================
# 🔮 PRÉVISION FUTURE (PROPHET)
# =================================================
st.markdown("<div class='section-title'> Prévisions futures de la consommation énergétique</div>", unsafe_allow_html=True)

df_prophet = df_pred[["date", "Real_Value"]].rename(columns={"date":"ds", "Real_Value":"y"})
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_model.fit(df_prophet)

future_periods = st.slider(" Sélectionner l’horizon de prévision (en mois) :", 1, 36, 12)
future = prophet_model.make_future_dataframe(periods=future_periods, freq='M')
forecast = prophet_model.predict(future)

fig_forecast = plot_plotly(prophet_model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

with st.expander(" Table des prévisions"):
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_periods))

st.success(" Tableau de bord chargé avec succès !")
