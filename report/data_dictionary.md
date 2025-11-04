Dataset: maroc_monthly_2000_2023.csv (Gold layer)

Source: Derived from maroc_energy_weather_1990_2023.csv (Silver layer)
Period: 2000 – 2023 (monthly data)
Description: Synthetic but realistic dataset simulating Morocco’s monthly electricity consumption based on energy mix and temperature trends.

| Column                        | Type  | Description                                                                            |
| ----------------------------- | ----- | -------------------------------------------------------------------------------------- |
| `year`                        | int   | Year of observation (2000–2023).                                                       |
| `month`                       | int   | Month number (1–12).                                                                   |
| `fossil_pct`                  | float | Percentage of electricity from fossil fuels.                                           |
| `oil_pct`                     | float | Percentage from oil-based sources.                                                     |
| `renewables_kWh`              | float | Production from non-hydro renewable sources (kWh).                                     |
| `temperature`                 | float | Average temperature for that month (°C).                                               |
| `electricity_consumption_GWh` | float | Total estimated electricity consumption (GWh), scaled to Morocco’s real yearly totals. |
