import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Configuración de Streamlit
st.set_page_config(layout="wide")

# Función para extracción de datos históricos
def extract_historical_data(ticker, start_date='2022-01-01'):
    df = yf.download(ticker, start=start_date, progress=False)
    df.reset_index(inplace=True)
    df['Crypto'] = ticker.split('-')[0]
    df = df[['Date', 'Close', 'Crypto']]
    df.rename(columns={'Date': 'Timestamp', 'Close': 'Actual Price'}, inplace=True)
    return df

# Transformación de datos
def transform_data(df):
    df['Highest 1H'] = df['Actual Price'].rolling(window=6).max()
    df['Lower 1H'] = df['Actual Price'].rolling(window=6).min()
    df['AVG Price'] = df['Actual Price'].rolling(window=6).mean()
    df['24hr_Change'] = df['Actual Price'].pct_change(periods=1).fillna(0) * 100
    df['Signal'] = df['24hr_Change'].apply(lambda x: 'B' if x > 0 else 'S')
    return df

# Pronóstico con Regresión Polinómica
def forecast_polynomial(df):
    df['Time_Index'] = np.arange(len(df))
    X = df[['Time_Index']]
    y = df['Actual Price']

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    forecast_index = np.array([[len(df) + i] for i in range(1, 31)])
    forecast_poly = poly.transform(forecast_index)
    forecast_prices = model.predict(forecast_poly)

    return pd.Series(forecast_prices.ravel()), model

# Pronóstico con ARIMA
def forecast_arima(df):
    model = ARIMA(df['Actual Price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return pd.Series(forecast.values.ravel()), model_fit

# Pronóstico con SARIMA
def forecast_sarima(df):
    model = SARIMAX(df['Actual Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    return pd.Series(forecast.values.ravel()), model_fit

# Visualización de los datos históricos
def plot_historical_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(df['Timestamp'], df['Highest 1H'], label='Highest 1H', linestyle='--', color='green')
    plt.plot(df['Timestamp'], df['Lower 1H'], label='Lower 1H', linestyle='--', color='red')
    plt.plot(df['Timestamp'], df['AVG Price'], label='AVG Price', linestyle=':', color='orange')
    plt.title('Historical Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Visualización del pronóstico individual
def plot_forecast(forecast, df, model_name):
    future_dates = [df['Timestamp'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, 31)]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(future_dates, forecast, label=f'{model_name} Forecast', linestyle='--', color='purple')
    plt.title(f'{model_name} Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Recomendación de compra o venta
def generate_recommendation(selected_date, forecasts):
    avg_forecast = np.mean([forecasts['Polynomial'][selected_date],
                            forecasts['ARIMA'][selected_date],
                            forecasts['SARIMA'][selected_date]])
    if avg_forecast > forecasts['Polynomial'][selected_date - 1]:
        return "Compra: Se espera una tendencia al alza."
    else:
        return "Venta: Se espera una tendencia a la baja."

# Streamlit App
def streamlit_app():
    st.title("Plataforma de Previsión de Operaciones con Yahoo Finanzas")

    cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'DOGE-USD']
    selected_crypto = st.selectbox("Selecciona la Criptomoneda", cryptos, index=0)

    data = extract_historical_data(selected_crypto, start_date='2022-01-01')
    transformed_data = transform_data(data)

    st.write("### Datos Históricos")
    plot_historical_data(transformed_data)

    st.write("### Todos los datos")
    st.dataframe(transformed_data, use_container_width=True)

    # Pronósticos
    poly_forecast, poly_model = forecast_polynomial(transformed_data)
    arima_forecast, arima_model = forecast_arima(transformed_data)
    sarima_forecast, sarima_model = forecast_sarima(transformed_data)

    # Gráficos y tablas para cada modelo
    st.write("### Regresión Polinómica")
    plot_forecast(poly_forecast, transformed_data, "Polynomial Regression")
    st.dataframe(pd.DataFrame({'Polynomial Forecast': poly_forecast}), use_container_width=True)

    st.write("### ARIMA Pronóstico")
    plot_forecast(arima_forecast, transformed_data, "ARIMA")
    st.dataframe(pd.DataFrame({'ARIMA Forecast': arima_forecast}), use_container_width=True)

    st.write("### SARIMA Pronóstico")
    plot_forecast(sarima_forecast, transformed_data, "SARIMA")
    st.dataframe(pd.DataFrame({'SARIMA Forecast': sarima_forecast}), use_container_width=True)

    # Recomendación personalizada
    st.write("### Recomendación Personalizada")
    selected_date = st.slider("Selecciona un día dentro del rango pronosticado", min_value=1, max_value=30)
    forecasts = {
        'Polynomial': poly_forecast,
        'ARIMA': arima_forecast,
        'SARIMA': sarima_forecast
    }
    recommendation = generate_recommendation(selected_date, forecasts)
    st.write(f"Recomendación para el día {selected_date}: **{recommendation}**")

# Pie de página
st.markdown("""
---
*EJERCICIO PRÁCTICO 3*
## **INTEGRANTES**
* Natasha Villacis
* Alexander Haro
* Darwin Ipiales
* Kempis Guerrero.
""")

if __name__ == '__main__':
    streamlit_app()
