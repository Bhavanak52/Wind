from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class WindSelectionEnv:
    """Offline RL environment where the agent selects the safest expert model at each step."""

    def __init__(self, data_df: pd.DataFrame, expert_predictions: np.ndarray, penalty_factor: float = 5.0):
        self.data_df = data_df.reset_index(drop=True).copy()
        self.expert_predictions = np.asarray(expert_predictions, dtype=np.float32)
        self.penalty_factor = penalty_factor
        self.n_steps = len(self.data_df)
        self.action_size = self.expert_predictions.shape[1]
        self.state_size = 8
        self.current_step = 0
        self.previous_errors = np.zeros(self.action_size, dtype=np.float32)
        # Normalize state channels to prevent high-magnitude pressure values from dominating Q-learning.
        self._state_means, self._state_stds = self._compute_state_normalization_stats()
        self._error_scale = max(float(self.data_df["wind_speed_40m"].std(skipna=True)), 1e-6)

    def _compute_state_normalization_stats(self) -> tuple[np.ndarray, np.ndarray]:
        state_cols = ["imf_trend", "imf_osc", "imf_noise", "air_pressure_40m", "temperature_40m"]
        means = []
        stds = []
        for col in state_cols:
            col_values = pd.to_numeric(self.data_df[col], errors="coerce")
            mean = float(col_values.mean(skipna=True))
            std = float(col_values.std(skipna=True))
            means.append(mean if np.isfinite(mean) else 0.0)
            stds.append(std if (np.isfinite(std) and std > 1e-6) else 1.0)
        return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)

    def _normalize_value(self, value: float, idx: int) -> float:
        return float((value - self._state_means[idx]) / self._state_stds[idx])

    def _get_state(self) -> np.ndarray:
        row = self.data_df.iloc[self.current_step]
        trend = self._normalize_value(float(row["imf_trend"]), 0)
        osc = self._normalize_value(float(row["imf_osc"]), 1)
        noise = self._normalize_value(float(row["imf_noise"]), 2)
        pressure = self._normalize_value(float(row["air_pressure_40m"]), 3)
        temperature = self._normalize_value(float(row["temperature_40m"]), 4)
        normalized_errors = (self.previous_errors / self._error_scale).tolist()
        return np.asarray(
            [
                trend,
                osc,
                noise,
                *normalized_errors,
                pressure,
                temperature,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.previous_errors = np.zeros(self.action_size, dtype=np.float32)
        return self._get_state()

    def step(self, action: int) -> tuple[Optional[np.ndarray], float, bool, dict]:
        row = self.data_df.iloc[self.current_step]
        true_value = float(row["wind_speed_40m"])
        expert_row = self.expert_predictions[self.current_step]
        selected_prediction = float(expert_row[action])

        errors = np.abs(expert_row - true_value).astype(np.float32)
        absolute_error = abs(selected_prediction - true_value)
        under_prediction = selected_prediction < true_value
        reward = -absolute_error * (self.penalty_factor if under_prediction else 1.0)

        info = {
            "timestamp": row["timestamp"],
            "target": true_value,
            "selected_prediction": selected_prediction,
            "all_predictions": expert_row.copy(),
            "action": int(action),
            "under_prediction": bool(under_prediction),
            "absolute_error": float(absolute_error),
        }

        self.previous_errors = errors
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_state = None if done else self._get_state()
        return next_state, float(reward), done, info
