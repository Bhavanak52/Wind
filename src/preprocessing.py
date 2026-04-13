from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


REQUIRED_COLUMNS = {
    "air pressure at 40m (Pa)": "air_pressure_40m",
    "temperature at 40m (C)": "temperature_40m",
    "wind direction at 40m (deg)": "wind_direction_40m",
    "wind speed at 100m (m/s)": "wind_speed_100m",
    "wind speed at 120m (m/s)": "wind_speed_120m",
    "wind speed at 80m (m/s)": "wind_speed_80m",
    "wind speed at 40m (m/s)": "wind_speed_40m",
}

DEFAULT_FEATURE_COLUMNS = [
    "imf_trend",
    "imf_osc",
    "imf_noise",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_40m",
    "air_pressure_40m",
    "temperature_40m",
]
DEFAULT_TARGET_COLUMN = "wind_speed_40m"
DEFAULT_NUM_MODES = 5
DEFAULT_LAG_STEPS = 12
DEFAULT_STATS_WINDOW = 12


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    raw_df: pd.DataFrame
    feature_scaler: MinMaxScaler
    feature_columns: List[str]
    target_column: str
    window_size: int
    mode_columns: List[str]
    num_modes: int
    lag_steps: int
    stats_window: int


@dataclass
class InferenceData:
    X: np.ndarray
    y: np.ndarray
    metadata_df: pd.DataFrame
    raw_df: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    window_size: int
    mode_columns: List[str]
    num_modes: int


def load_or_generate_wind_data(csv_path: str | Path, rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Load the provided dataset or generate a synthetic 5-minute wind series if missing."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="5min")
    daily_wave = 2.5 + 1.2 * np.sin(np.linspace(0, 40 * np.pi, rows))
    weather_wave = 0.8 * np.sin(np.linspace(0, 8 * np.pi, rows) + 1.5)
    turbulence = rng.normal(0.0, 0.25, rows)
    wind_40 = np.clip(daily_wave + weather_wave + turbulence, 0.1, None)

    df = pd.DataFrame(
        {
            "Year": timestamps.year,
            "Month": timestamps.month,
            "Day": timestamps.day,
            "Hour": timestamps.hour,
            "Minute": timestamps.minute,
            "air pressure at 40m (Pa)": 92100 + 120 * np.sin(np.linspace(0, 6 * np.pi, rows)) + rng.normal(0, 10, rows),
            "temperature at 40m (C)": 18 + 6 * np.sin(np.linspace(0, 6 * np.pi, rows) - 0.5) + rng.normal(0, 0.4, rows),
            "wind direction at 40m (deg)": (180 + 80 * np.sin(np.linspace(0, 15 * np.pi, rows)) + rng.normal(0, 5, rows)) % 360,
            "wind speed at 100m (m/s)": wind_40 * 1.12 + rng.normal(0, 0.08, rows),
            "wind speed at 120m (m/s)": wind_40 * 1.16 + rng.normal(0, 0.08, rows),
            "wind speed at 80m (m/s)": wind_40 * 1.08 + rng.normal(0, 0.06, rows),
            "wind speed at 40m (m/s)": wind_40,
        }
    )
    df.to_csv(csv_path, index=False)
    return df


def _rename_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy().rename(columns=REQUIRED_COLUMNS)
    numeric_columns = list(REQUIRED_COLUMNS.values())
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df["timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[numeric_columns] = df[numeric_columns].interpolate(method="linear").ffill().bfill()

    # IQR clipping limits the impact of sensor spikes while preserving trend shape.
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=high)

    return df


def _safe_entropy(values: np.ndarray, bins: int = 10) -> float:
    hist, _ = np.histogram(values, bins=bins)
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs /= total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 2)).apply(
        lambda x: _safe_entropy(np.asarray(x, dtype=np.float64), bins=bins),
        raw=True,
    )


def decompose_wind_signal_ewt(signal: pd.Series, num_modes: int = DEFAULT_NUM_MODES) -> Dict[str, np.ndarray]:
    """EWT-style decomposition into multiple frequency modes + residual correction."""
    if num_modes < 3:
        raise ValueError("num_modes must be at least 3 for trend/oscillation/noise separation.")

    values = signal.astype(float).to_numpy()

    try:
        import pywt

        wavelet = "db4"
        max_level = pywt.dwt_max_level(data_len=len(values), filter_len=pywt.Wavelet(wavelet).dec_len)
        level = min(max(1, num_modes - 1), max_level)
        coeffs = pywt.wavedec(values, wavelet=wavelet, level=level)

        components: List[np.ndarray] = []

        approx_only = [np.zeros_like(c) for c in coeffs]
        approx_only[0] = coeffs[0]
        approx_component = pywt.waverec(approx_only, wavelet=wavelet)[: len(values)]
        components.append(approx_component)

        for detail_idx in range(1, len(coeffs)):
            detail_only = [np.zeros_like(c) for c in coeffs]
            detail_only[detail_idx] = coeffs[detail_idx]
            detail_component = pywt.waverec(detail_only, wavelet=wavelet)[: len(values)]
            components.append(detail_component)

        while len(components) < num_modes:
            components.append(np.zeros_like(values, dtype=np.float64))

        components = components[:num_modes]
        reconstructed = np.sum(np.vstack(components), axis=0)
        residual = values - reconstructed
        components[0] = components[0] + residual

        mode_payload = {f"mode_{idx + 1}": comp.astype(np.float32) for idx, comp in enumerate(components)}
        return mode_payload

    except Exception:
        try:
            import pywt

            widths = np.arange(1, min(128, len(values) // 2 + 1))
            coeff = pywt.cwt(values, widths, "morl", method="conv")[0]
            band_edges = np.linspace(0, coeff.shape[0], num=num_modes + 1, dtype=int)
            components = []
            for idx in range(num_modes):
                lo, hi = band_edges[idx], band_edges[idx + 1]
                band_signal = np.mean(np.abs(coeff[lo:hi]), axis=0) if hi > lo else np.zeros_like(values)
                components.append(band_signal)

            total = np.sum(np.vstack(components), axis=0)
            total[total == 0] = 1.0
            scale = values / total
            components = [(comp * scale).astype(np.float32) for comp in components]

            reconstructed = np.sum(np.vstack(components), axis=0)
            components[0] = components[0] + (values - reconstructed)
            return {f"mode_{idx + 1}": comp for idx, comp in enumerate(components)}

        except Exception:
            print("[WARNING] EWT decomposition failed, using rolling window fallback")
            components = []
            spans = np.linspace(6, 36, num=num_modes, dtype=int)
            prev = np.zeros_like(values, dtype=np.float64)
            for span in spans:
                smooth = pd.Series(values).rolling(window=int(span), min_periods=1).mean().to_numpy()
                comp = smooth - prev
                components.append(comp)
                prev = smooth

            reconstructed = np.sum(np.vstack(components), axis=0)
            components[0] = components[0] + (values - reconstructed)
            return {f"mode_{idx + 1}": comp.astype(np.float32) for idx, comp in enumerate(components)}


def decompose_wind_signal_emd_proxy(signal: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible 3-component view: trend, aggregate oscillation, high-frequency noise."""
    mode_payload = decompose_wind_signal_ewt(signal=signal, num_modes=DEFAULT_NUM_MODES)
    mode_keys = list(mode_payload.keys())
    trend = mode_payload[mode_keys[0]]
    noise = mode_payload[mode_keys[-1]]
    if len(mode_keys) > 2:
        mids = [mode_payload[k] for k in mode_keys[1:-1]]
        osc = np.mean(np.vstack(mids), axis=0).astype(np.float32)
    else:
        osc = np.zeros_like(trend)
    return trend, osc, noise


def build_fused_features(
    df: pd.DataFrame,
    mode_columns: List[str],
    target_column: str,
    lag_steps: int = DEFAULT_LAG_STEPS,
    stats_window: int = DEFAULT_STATS_WINDOW,
) -> tuple[pd.DataFrame, List[str]]:
    feature_df = pd.DataFrame(index=df.index)

    base_exogenous = [
        "wind_speed_80m",
        "wind_speed_100m",
        "wind_speed_120m",
        "wind_direction_40m",
        "air_pressure_40m",
        "temperature_40m",
    ]

    for col in base_exogenous:
        feature_df[col] = df[col].astype(float)

    for mode_col in mode_columns:
        mode_series = df[mode_col].astype(float)
        feature_df[mode_col] = mode_series
        feature_df[f"{mode_col}_mean"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).mean()
        feature_df[f"{mode_col}_std"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).std()
        feature_df[f"{mode_col}_min"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).min()
        feature_df[f"{mode_col}_max"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).max()
        feature_df[f"{mode_col}_median"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).median()
        feature_df[f"{mode_col}_entropy"] = _rolling_entropy(mode_series, window=stats_window, bins=10)
        feature_df[f"{mode_col}_skew"] = mode_series.rolling(stats_window, min_periods=max(3, stats_window // 2)).skew()

    for lag in range(1, lag_steps + 1):
        feature_df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return feature_df, feature_df.columns.tolist()


def create_sliding_windows(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, rows = [], [], []

    for idx in range(window_size, len(feature_matrix)):
        X.append(feature_matrix[idx - window_size : idx])
        y.append(targets[idx])
        rows.append(metadata.iloc[idx].to_dict())

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    meta_df = pd.DataFrame(rows)
    return X_array, y_array, meta_df


def prepare_inference_data(
    csv_path: str | Path,
    window_size: int = 12,
    feature_scaler: Optional[MinMaxScaler] = None,
    feature_columns: Optional[List[str]] = None,
    num_modes: int = DEFAULT_NUM_MODES,
    lag_steps: int = DEFAULT_LAG_STEPS,
    stats_window: int = DEFAULT_STATS_WINDOW,
    max_rows: Optional[int] = None,
) -> InferenceData:
    raw_df = load_or_generate_wind_data(csv_path)
    df = _rename_and_clean_columns(raw_df)

    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    mode_payload = decompose_wind_signal_ewt(df[DEFAULT_TARGET_COLUMN], num_modes=num_modes)
    mode_columns = []
    for name, values in mode_payload.items():
        col_name = f"ewt_{name}"
        df[col_name] = values
        mode_columns.append(col_name)

    trend, osc, noise = decompose_wind_signal_emd_proxy(df[DEFAULT_TARGET_COLUMN])
    df["imf_trend"] = trend
    df["imf_osc"] = osc
    df["imf_noise"] = noise

    fused_df, fused_feature_columns = build_fused_features(
        df=df,
        mode_columns=mode_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        lag_steps=lag_steps,
        stats_window=stats_window,
    )

    feature_columns = list(feature_columns or fused_feature_columns)
    metadata_columns = [
        "timestamp",
        "imf_trend",
        "imf_osc",
        "imf_noise",
        "air_pressure_40m",
        "temperature_40m",
        DEFAULT_TARGET_COLUMN,
    ]

    scaler = feature_scaler or MinMaxScaler()
    if feature_scaler is None:
        scaler.fit(fused_df[feature_columns])
    scaled_features = scaler.transform(fused_df[feature_columns])

    X, y, meta_df = create_sliding_windows(
        scaled_features,
        df[DEFAULT_TARGET_COLUMN].to_numpy(dtype=np.float32),
        df[metadata_columns],
        window_size,
    )

    if len(X) == 0:
        raise ValueError(f"Need at least {window_size + 1} rows in the CSV to create rolling prediction windows.")

    return InferenceData(
        X=X,
        y=y,
        metadata_df=meta_df.reset_index(drop=True),
        raw_df=df,
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        window_size=window_size,
        mode_columns=mode_columns,
        num_modes=num_modes,
    )


def prepare_single_forecast_data(
    csv_path: str | Path,
    window_size: int = 12,
    feature_scaler: Optional[MinMaxScaler] = None,
    feature_columns: Optional[List[str]] = None,
    num_modes: int = DEFAULT_NUM_MODES,
    lag_steps: int = DEFAULT_LAG_STEPS,
    stats_window: int = DEFAULT_STATS_WINDOW,
    max_rows: Optional[int] = None,
) -> InferenceData:
    raw_df = load_or_generate_wind_data(csv_path)
    df = _rename_and_clean_columns(raw_df)

    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    if len(df) < window_size:
        raise ValueError(f"Need at least {window_size} rows in the CSV to forecast the next 5-minute step.")

    mode_payload = decompose_wind_signal_ewt(df[DEFAULT_TARGET_COLUMN], num_modes=num_modes)
    mode_columns = []
    for name, values in mode_payload.items():
        col_name = f"ewt_{name}"
        df[col_name] = values
        mode_columns.append(col_name)

    trend, osc, noise = decompose_wind_signal_emd_proxy(df[DEFAULT_TARGET_COLUMN])
    df["imf_trend"] = trend
    df["imf_osc"] = osc
    df["imf_noise"] = noise

    fused_df, fused_feature_columns = build_fused_features(
        df=df,
        mode_columns=mode_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        lag_steps=lag_steps,
        stats_window=stats_window,
    )

    feature_columns = list(feature_columns or fused_feature_columns)
    scaler = feature_scaler or MinMaxScaler()
    if feature_scaler is None:
        scaler.fit(fused_df[feature_columns])
    scaled_features = scaler.transform(fused_df[feature_columns])

    X = np.asarray([scaled_features[-window_size:]], dtype=np.float32)
    last_row = df.iloc[-1]
    forecast_timestamp = last_row["timestamp"] + pd.Timedelta(minutes=5)
    metadata_df = pd.DataFrame(
        [
            {
                "timestamp": forecast_timestamp,
                "forecast_timestamp": forecast_timestamp,
                "input_window_start": df.iloc[-window_size]["timestamp"],
                "input_window_end": last_row["timestamp"],
                "last_known_wind_speed_40m": last_row[DEFAULT_TARGET_COLUMN],
                "imf_trend": last_row["imf_trend"],
                "imf_osc": last_row["imf_osc"],
                "imf_noise": last_row["imf_noise"],
                "air_pressure_40m": last_row["air_pressure_40m"],
                "temperature_40m": last_row["temperature_40m"],
                DEFAULT_TARGET_COLUMN: np.nan,
            }
        ]
    )

    return InferenceData(
        X=X,
        y=np.asarray([], dtype=np.float32),
        metadata_df=metadata_df.reset_index(drop=True),
        raw_df=df,
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        window_size=window_size,
        mode_columns=mode_columns,
        num_modes=num_modes,
    )


def prepare_datasets(
    csv_path: str | Path,
    window_size: int = 12,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_modes: int = DEFAULT_NUM_MODES,
    lag_steps: int = DEFAULT_LAG_STEPS,
    stats_window: int = DEFAULT_STATS_WINDOW,
    max_rows: Optional[int] = None,
) -> PreparedData:
    raw_df = load_or_generate_wind_data(csv_path)
    df = _rename_and_clean_columns(raw_df)

    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    mode_payload = decompose_wind_signal_ewt(df[DEFAULT_TARGET_COLUMN], num_modes=num_modes)
    mode_columns = []
    for name, values in mode_payload.items():
        col_name = f"ewt_{name}"
        df[col_name] = values
        mode_columns.append(col_name)

    trend, osc, noise = decompose_wind_signal_emd_proxy(df[DEFAULT_TARGET_COLUMN])
    df["imf_trend"] = trend
    df["imf_osc"] = osc
    df["imf_noise"] = noise

    fused_df, feature_columns = build_fused_features(
        df=df,
        mode_columns=mode_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        lag_steps=lag_steps,
        stats_window=stats_window,
    )

    target_column = DEFAULT_TARGET_COLUMN
    metadata_columns = [
        "timestamp",
        "imf_trend",
        "imf_osc",
        "imf_noise",
        "air_pressure_40m",
        "temperature_40m",
        target_column,
    ]

    scaler = MinMaxScaler()
    train_row_end = max(int(len(df) * train_ratio), window_size + 1)
    scaler.fit(fused_df.loc[: train_row_end - 1, feature_columns])
    scaled_features = scaler.transform(fused_df[feature_columns])

    X, y, meta_df = create_sliding_windows(
        scaled_features,
        df[target_column].to_numpy(dtype=np.float32),
        df[metadata_columns],
        window_size,
    )

    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    return PreparedData(
        X_train=X[:train_end],
        y_train=y[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
        train_df=meta_df.iloc[:train_end].reset_index(drop=True),
        val_df=meta_df.iloc[train_end:val_end].reset_index(drop=True),
        test_df=meta_df.iloc[val_end:].reset_index(drop=True),
        raw_df=df,
        feature_scaler=scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        mode_columns=mode_columns,
        num_modes=num_modes,
        lag_steps=lag_steps,
        stats_window=stats_window,
    )


def save_preprocessing_artifacts(prepared: PreparedData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(
            {
                "feature_scaler": prepared.feature_scaler,
                "feature_columns": prepared.feature_columns,
                "target_column": prepared.target_column,
                "window_size": prepared.window_size,
                "mode_columns": prepared.mode_columns,
                "num_modes": prepared.num_modes,
                "lag_steps": prepared.lag_steps,
                "stats_window": prepared.stats_window,
            },
            handle,
        )
    return output_path


def load_preprocessing_artifacts(artifact_path: str | Path) -> dict:
    artifact_path = Path(artifact_path)
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


if __name__ == "__main__":
    prepared = prepare_datasets("wind_data.csv")
    print("X_train:", prepared.X_train.shape)
    print("X_val:", prepared.X_val.shape)
    print("X_test:", prepared.X_test.shape)
