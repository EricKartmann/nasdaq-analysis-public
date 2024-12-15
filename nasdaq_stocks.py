import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="NASDAQ Top Companies", page_icon="📈", layout="wide")
st.title("📊 Top 7 Empresas del NASDAQ - Análisis 4H")

# Lista de símbolos de las principales empresas del NASDAQ
top_companies = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.'
}

def resample_to_4h(data):
    """Convierte los datos a intervalos de 4 horas"""
    data_4h = data.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return data_4h

def format_volume(volume):
    """Formatea el volumen en K, M o B según su tamaño"""
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    return str(volume)

def calculate_rsi(data, periods=14):
    """Calcula el RSI (Relative Strength Index)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calcula el MACD (Moving Average Convergence Divergence)"""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1], macd, signal

def calculate_bollinger_bands(data, window=20):
    """Calcula las Bandas de Bollinger"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

def calculate_stochastic(data, k_window=14, d_window=3):
    """Calcula el Oscilador Estocástico"""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return k.iloc[-1], d.iloc[-1]

def get_premarket_data(symbol):
    """Obtiene datos del pre-mercado"""
    try:
        stock = yf.Ticker(symbol)
        # Obtener datos del pre-mercado (últimas 4 horas antes de la apertura)
        premarket = stock.history(period='1d', interval='1m', prepost=True)
        if not premarket.empty:
            # Filtrar solo datos del pre-mercado
            current_hour = datetime.now().hour
            if current_hour < 9:  # Si estamos antes de la apertura del mercado
                premarket = premarket[premarket.index.hour < 9]
                if not premarket.empty:
                    return {
                        'price': premarket['Close'].iloc[-1],
                        'change': ((premarket['Close'].iloc[-1] - premarket['Open'].iloc[0]) / premarket['Open'].iloc[0]) * 100,
                        'volume': premarket['Volume'].sum()
                    }
    except Exception:
        pass
    return None

def predict_next_day_movement(hist_4h, macd_line, signal_line, premarket_data=None):
    """Predice la tendencia para el próximo día basado en patrones de 4 horas y datos del pre-mercado"""
    # Análisis de tendencia reciente (últimas 24 horas = 6 períodos de 4h)
    recent_trend = hist_4h['Close'].tail(6).pct_change().mean() * 100
    
    # Momentum del MACD
    macd_momentum = (macd_line.tail(6) - signal_line.tail(6)).mean()
    
    # Volatilidad reciente
    recent_volatility = hist_4h['Close'].tail(6).pct_change().std() * 100
    
    # Fuerza de la tendencia base
    trend_strength = abs(recent_trend) / recent_volatility if recent_volatility != 0 else 0
    
    # Ajustar predicción con datos del pre-mercado si están disponibles
    if premarket_data:
        premarket_trend = premarket_data['change']
        
        # Ajustar la tendencia base con el pre-mercado
        if abs(premarket_trend) > 1:  # Si hay un movimiento significativo en el pre-mercado
            if premarket_trend > 0:
                recent_trend = max(recent_trend, 0) + premarket_trend/2  # Dar más peso al pre-mercado
            else:
                recent_trend = min(recent_trend, 0) + premarket_trend/2
            
            # Ajustar la fuerza de la tendencia
            trend_strength = trend_strength * 1.5 if np.sign(recent_trend) == np.sign(premarket_trend) else trend_strength * 0.8
    
    prediction = {
        'Tendencia': "ALCISTA" if recent_trend > 0 else "BAJISTA",
        'Fuerza': "FUERTE" if trend_strength > 1 else "MODERADA" if trend_strength > 0.5 else "DÉBIL",
        'Volatilidad': f"{recent_volatility:.2f}%",
        'Confianza': "ALTA" if trend_strength > 1.5 else "MEDIA" if trend_strength > 0.75 else "BAJA"
    }
    
    # Añadir información del pre-mercado si está disponible
    if premarket_data:
        prediction['Premarket'] = f"{premarket_data['change']:.2f}%"
    
    return prediction

def get_advanced_signals(price, upper_bb, lower_bb, macd, macd_signal, k, d):
    """Analiza señales de indicadores avanzados"""
    signals = []
    
    # Señales de Bandas de Bollinger
    if price > upper_bb:
        signals.append(("VENTA", "Precio sobre banda superior de Bollinger"))
    elif price < lower_bb:
        signals.append(("COMPRA", "Precio bajo banda inferior de Bollinger"))
    
    # Señales de MACD
    if macd > macd_signal and macd > 0:
        signals.append(("COMPRA", "Cruce alcista del MACD"))
    elif macd < macd_signal and macd < 0:
        signals.append(("VENTA", "Cruce bajista del MACD"))
    
    # Señales del Estocástico
    if k > 80 and d > 80:
        signals.append(("VENTA", "Estocástico en zona de sobrecompra"))
    elif k < 20 and d < 20:
        signals.append(("COMPRA", "Estocástico en zona de sobreventa"))
    
    return signals

def get_recommendation(price_change, rsi, volume_change, sma_ratio, price, upper_bb, lower_bb, macd, macd_signal, k, d):
    """Genera una recomendación basada en todos los indicadores técnicos"""
    signals = []
    
    # Señales básicas
    if price_change > 2:
        signals.append(("VENTA", "Subida fuerte de precio"))
    elif price_change < -2:
        signals.append(("COMPRA", "Caída significativa de precio"))
        
    if rsi > 70:
        signals.append(("VENTA", "RSI en sobrecompra"))
    elif rsi < 30:
        signals.append(("COMPRA", "RSI en sobreventa"))
        
    if volume_change > 50:
        signals.append(("COMPRA" if price_change > 0 else "VENTA", "Alto volumen de operaciones"))
        
    if sma_ratio > 1.05:
        signals.append(("COMPRA", "Tendencia alcista fuerte"))
    elif sma_ratio < 0.95:
        signals.append(("VENTA", "Tendencia bajista fuerte"))
    
    # Añadir señales avanzadas
    signals.extend(get_advanced_signals(price, upper_bb, lower_bb, macd, macd_signal, k, d))
    
    if not signals:
        return "MANTENER", "Sin señales claras de trading"
    
    # Contar señales de compra y venta
    buy_signals = len([s for s in signals if s[0] == "COMPRA"])
    sell_signals = len([s for s in signals if s[0] == "VENTA"])
    
    # Generar razón detallada
    all_reasons = [s[1] for s in signals]
    detailed_reason = " | ".join(all_reasons[:3])
    
    if buy_signals > sell_signals:
        return "COMPRA", detailed_reason
    elif sell_signals > buy_signals:
        return "VENTA", detailed_reason
    else:
        return "MANTENER", "Señales mixtas: " + detailed_reason

def get_stock_data(symbols):
    data = []
    for symbol in symbols:
        try:
            # Obtener datos del pre-mercado
            premarket_data = get_premarket_data(symbol)
            
            stock = yf.Ticker(symbol)
            # Obtener datos de 7 días con intervalos de 1 hora
            hist = stock.history(period='7d', interval='1h')
            
            if not hist.empty and len(hist) > 26:
                # Convertir a intervalos de 4 horas
                hist_4h = resample_to_4h(hist)
                
                current_price = hist_4h['Close'].iloc[-1]
                previous_close = hist_4h['Close'].iloc[-2]
                
                # Cálculos básicos
                price_change = ((current_price - previous_close) / previous_close) * 100
                rsi = calculate_rsi(hist_4h).iloc[-1]
                volume_change = ((hist_4h['Volume'].iloc[-1] - hist_4h['Volume'].mean()) / hist_4h['Volume'].mean()) * 100
                sma_20 = hist_4h['Close'].rolling(window=20).mean().iloc[-1]
                sma_ratio = current_price / sma_20
                
                # Indicadores avanzados
                macd, macd_signal, macd_line, signal_line = calculate_macd(hist_4h)
                upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(hist_4h)
                k, d = calculate_stochastic(hist_4h)
                
                # Predicción para el próximo día incluyendo datos del pre-mercado
                prediction = predict_next_day_movement(hist_4h, macd_line, signal_line, premarket_data)
                
                # Obtener recomendación actual
                current_rec, current_reason = get_recommendation(
                    price_change, rsi, volume_change, sma_ratio,
                    current_price, upper_bb, lower_bb, macd, macd_signal, k, d
                )
                
                # Combinar la predicción con la recomendación actual
                if prediction['Tendencia'] == "ALCISTA":
                    main_recommendation = "COMPRA"
                elif prediction['Tendencia'] == "BAJISTA":
                    main_recommendation = "VENTA"
                else:
                    main_recommendation = "MANTENER"
                
                # Incluir datos del pre-mercado en la razón si están disponibles
                premarket_info = f" | Premarket: {prediction.get('Premarket', 'No disponible')}" if 'Premarket' in prediction else ""
                recommendation_reason = f"{prediction['Tendencia']} ({prediction['Fuerza']}) | Confianza: {prediction['Confianza']}{premarket_info} | {current_reason}"
                
                # Calcular volumen promedio
                daily_volume = hist_4h['Volume'].tail(6).mean()
                avg_volume = format_volume(int(daily_volume))
                
                data.append({
                    'Símbolo': symbol,
                    'Empresa': top_companies[symbol],
                    'Precio Actual ($)': current_price,
                    'Recomendación': main_recommendation,
                    'Razón': recommendation_reason,
                    'Cierre Anterior ($)': previous_close,
                    'Cambio (%)': f"{price_change:.2f}%",
                    'Volumen 24h': avg_volume,
                    'RSI': f"{rsi:.1f}",
                    'MACD': f"{macd:.2f}",
                    'Estocástico %K': f"{k:.1f}"
                })
            else:
                st.warning(f"No se pudieron obtener suficientes datos para {symbol}")
        except Exception as e:
            st.error(f"Error al obtener datos para {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

# Obtener y mostrar los datos
with st.spinner('Cargando datos del mercado...'):
    df = get_stock_data(top_companies.keys())

# Aplicar estilo a la tabla
if not df.empty:
    st.dataframe(
        df,
        column_config={
            "Símbolo": st.column_config.TextColumn("Símbolo", width="small"),
            "Empresa": st.column_config.TextColumn("Empresa", width="medium"),
            "Precio Actual ($)": st.column_config.NumberColumn(
                "Precio Actual ($)",
                format="$%.2f",
                help="Precio actual de la acción"
            ),
            "Recomendación": st.column_config.TextColumn(
                "Recomendación",
                help="Recomendación de trading basada en análisis técnico"
            ),
            "Razón": st.column_config.TextColumn(
                "Razón",
                help="Explicación de la recomendación"
            ),
            "Cierre Anterior ($)": st.column_config.NumberColumn(
                "Cierre Anterior ($)",
                format="$%.2f",
                help="Precio de cierre anterior (4h)"
            ),
            "Cambio (%)": st.column_config.TextColumn(
                "Cambio (%)",
                width="small",
                help="Cambio porcentual en las últimas 4h"
            ),
            "Volumen 24h": st.column_config.TextColumn(
                "Volumen 24h",
                help="Volumen promedio en las últimas 24 horas"
            ),
            "RSI": st.column_config.TextColumn(
                "RSI",
                help="Relative Strength Index (RSI) - Indicador de sobrecompra/sobreventa"
            ),
            "MACD": st.column_config.NumberColumn(
                "MACD",
                format="%.2f",
                help="Moving Average Convergence Divergence"
            ),
            "Estocástico %K": st.column_config.NumberColumn(
                "Estocástico %K",
                format="%.1f",
                help="Oscilador Estocástico %K"
            ),
        },
        hide_index=True,
    )
    
    # Agregar explicación de los indicadores
    with st.expander("ℹ️ Información sobre los Indicadores Técnicos (4H)"):
        st.markdown("""
        ### Indicadores Técnicos en Temporalidad 4H

        #### 1. RSI (Relative Strength Index)
        - Mide el momentum del precio en períodos de 4 horas
        - RSI > 70: Posible sobrecompra
        - RSI < 30: Posible sobreventa

        #### 2. MACD (Moving Average Convergence Divergence)
        - Indicador de tendencia adaptado a 4 horas
        - MACD positivo y creciente: Señal alcista
        - MACD negativo y decreciente: Señal bajista

        #### 3. Bandas de Bollinger
        - Miden la volatilidad en períodos de 4 horas
        - Precio cerca de la banda superior: Posible sobrecompra
        - Precio cerca de la banda inferior: Posible sobreventa

        #### 4. Oscilador Estocástico
        - Compara el precio de cierre con el rango de precios en 4 horas
        - %K > 80: Condición de sobrecompra
        - %K < 20: Condición de sobreventa

        #### 5. Predicción del Próximo Día
        - Basada en el análisis de patrones de las últimas 24 horas
        - Considera la tendencia, momentum y volatilidad
        - Niveles de confianza: ALTA, MEDIA, BAJA
        - Fuerza de la tendencia: FUERTE, MODERADA, DÉBIL

        **Nota**: Las predicciones son estimaciones basadas en análisis técnico y no garantizan resultados futuros.
        """)
else:
    st.error("No se pudieron obtener datos del mercado. Por favor, intenta de nuevo más tarde.")

# Mostrar última actualización
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Agregar botón de actualización
if st.button("🔄 Actualizar Datos"):
    st.rerun() 