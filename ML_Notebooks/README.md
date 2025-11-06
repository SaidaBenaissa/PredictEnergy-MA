# 🔌 Projet de Prévision de la Consommation Électrique au Maroc

**Système intelligent de prévision de la consommation électrique utilisant l'IA pour optimiser la planification énergétique au Maroc (2000-2023)**

---

## 📊 Résultats Clés

| Métrique | Prophet | LSTM (Gagnant) | Amélioration |
|----------|---------|----------------|--------------|
| **MAE** | 185.55 GWh | **140.28 GWh** | **+24.4%** 🎯 |
| **RMSE** | 245.10 GWh | **185.42 GWh** | **+24.3%** 📈 |
| **MAPE** | 5.81% | **4.39%** | **+24.5%** 🏆 |
| **R²** | -0.5422 | **0.2345** | **+143.2%** ⚡ |

**🎯 Performance moyenne : MAPE de 4.39% (Niveau Excellence)**

---

## 🚀 Fonctionnalités

- **📈 Prévisions mensuelles** de consommation électrique
- **🤖 Deux modèles comparés** : Prophet (Facebook) vs LSTM (TensorFlow)
- **🔮 Prévisions 12 mois** avec intervalles de confiance
- **📊 Dashboard interactif** pour l'analyse des résultats
- **⚡ API REST** pour l'intégration aux systèmes existants
- **🔍 Monitoring automatique** des performances

---

## 🧠 Modèles Implémentés

### 1. **Facebook Prophet** 📊
- **Algorithme** : Modèle additif avec composantes saisonnières
- **Features** : Temperature, mix énergétique, saisonnalité
- **Performance** : MAPE 5.81% (Très bon)

### 2. **LSTM Univarié** 🏆 (**MODÈLE GAGNANT**)
- **Architecture** : Réseau neuronal récurrent à mémoire longue
- **Séquences** : 12 mois d'historique
- **Performance** : MAPE 4.39% (Excellence)
- **Avantage** : Capture les patterns temporels complexes

---

## 📈 Résultats Détailés par Année

| Année | MAE LSTM | MAE Prophet | Gagnant | Avantage LSTM |
|-------|----------|-------------|---------|---------------|
| 2019 | 82.9 GWh | 104.5 GWh | **LSTM** 🏆 | +20.7% |
| 2020 | 51.9 GWh | 61.9 GWh | **LSTM** 🏆 | +16.2% |
| 2021 | 100.8 GWh | 117.8 GWh | **LSTM** 🏆 | +14.4% |
| 2022 | 237.8 GWh | 268.0 GWh | **LSTM** 🏆 | +11.3% |
| 2023 | 351.7 GWh | 388.2 GWh | **LSTM** 🏆 | +9.4% |

**🎯 LSTM gagne sur les 5 années de test !**

---

## 🛠️ Installation et Utilisation

### Prérequis
```bash
Python 3.8+
TensorFlow 2.12+
Prophet 1.1+