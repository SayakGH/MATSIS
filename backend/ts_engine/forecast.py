import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)

def prophet_forecast(df: pd.DataFrame, horizon: int, freq: str = "D") -> ToolResult:
    try:
        pdf = df[['timestamp', 'value']].rename(columns={"timestamp": "ds", "value": "y"})
        # Prophet requires timezone-naive datetimes — strip tz if present
        pdf['ds'] = pd.to_datetime(pdf['ds'], errors='coerce').dt.tz_localize(None)
        model = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True)
        model.fit(pdf)
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)
        chart_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        chart_data['timestamp'] = chart_data['ds'].astype(str)
        chart_data = chart_data.drop(columns=['ds']).to_dict('records')
        # Merge actuals
        actual_map = {str(r['timestamp']): r['value'] for _, r in df.iterrows()}
        for d in chart_data:
            d['value'] = actual_map.get(d['timestamp'])
        return ToolResult(
            task="forecast",
            raw_output=forecast.tail(horizon).to_dict('records'),
            chart_data=chart_data,
            metrics={"method": "prophet", "horizon": horizon}
        )
    except Exception as e:
        logger.error(f"Prophet forecast error: {e}")
        return ToolResult(task="forecast", raw_output={}, chart_data=[], metrics={}, error=str(e))

def arima_forecast(df: pd.DataFrame, horizon: int) -> ToolResult:
    try:
        model = ARIMA(df['value'], order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=horizon)
        import pandas as _pd
        future_dates = _pd.date_range(start=df['timestamp'].iloc[-1], periods=horizon + 1, freq='D')[1:]
        chart_data = [{"timestamp": str(d), "value": None, "yhat": float(v)} for d, v in zip(future_dates, forecast)]
        return ToolResult(
            task="forecast",
            raw_output=chart_data,
            chart_data=chart_data,
            metrics={"method": "arima", "horizon": horizon}
        )
    except Exception as e:
        logger.error(f"ARIMA forecast error: {e}")
        return ToolResult(task="forecast", raw_output={}, chart_data=[], metrics={}, error=str(e))

def lstm_forecast(df: pd.DataFrame, horizon: int) -> ToolResult:
    if len(df) < 200:
        logger.warning("Not enough data for LSTM (< 200 rows). Falling back to Prophet.")
        return prophet_forecast(df, horizon)
    try:
        from neuralprophet import NeuralProphet
        m = NeuralProphet()
        ndf = df[['timestamp', 'value']].rename(columns={"timestamp": "ds", "value": "y"})
        m.fit(ndf, freq="D")
        future = m.make_future_dataframe(ndf, periods=horizon, n_historic_predictions=True)
        forecast = m.predict(future)
        chart_data = [{"timestamp": str(r['ds']), "yhat": float(r.get('yhat1', 0))} for _, r in forecast.iterrows()]
        return ToolResult(
            task="forecast",
            raw_output=chart_data,
            chart_data=chart_data,
            metrics={"method": "lstm", "horizon": horizon}
        )
    except Exception as e:
        logger.error(f"LSTM forecast error: {e}")
        return ToolResult(task="forecast", raw_output={}, chart_data=[], metrics={}, error=str(e))
