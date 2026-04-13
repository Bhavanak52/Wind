from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.drl_agent import DQNAgent, evaluate_policy
from src.environment import WindSelectionEnv
from src.models import MODEL_REGISTRY
from src.preprocessing import load_preprocessing_artifacts, prepare_inference_data, prepare_single_forecast_data
from src.train import predict_model
from src.visualize import plot_actual_vs_predicted


MODEL_FILENAMES = {
    model_name: f"{model_name.replace('-', '_')}_model.pth" for model_name in MODEL_REGISTRY
}
MODEL_NAMES = list(MODEL_REGISTRY.keys())


def stage_log(stage: int, total: int, title: str) -> None:
    print(f"\n[Stage {stage}/{total}] {title}")


def _build_timestamped_name(prefix: str) -> str:
    return f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"


def _resolve_output_path(
    output_csv: str | Path | None,
    default_dir: Path,
    default_prefix: str,
    overwrite: bool,
) -> Path:
    if output_csv is None:
        candidate = default_dir / _build_timestamped_name(default_prefix)
    else:
        candidate = Path(output_csv)

    candidate.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix or ".csv"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    fallback = candidate.with_name(f"{stem}_{timestamp}{suffix}")
    if not fallback.exists():
        return fallback

    counter = 1
    while True:
        versioned = candidate.with_name(f"{stem}_{timestamp}_{counter}{suffix}")
        if not versioned.exists():
            return versioned
        counter += 1


def _print_row_wise_forecast(row: pd.Series) -> None:
    ordered_keys = [
        "input_window_start",
        "input_window_end",
        "forecast_timestamp",
        "last_known_wind_speed_40m",
        "pred_lstm",
        "pred_cnn_lstm",
        "pred_cnn_gru",
        "selected_action",
        "selected_model",
        "pred_drl_hybrid",
    ]
    print("Forecast details (row-wise):")
    for key in ordered_keys:
        value = row[key]
        if isinstance(value, (float, np.floating)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def load_expert_models(model_dir: str | Path, input_size: int, device: torch.device) -> Dict[str, torch.nn.Module]:
    model_dir = Path(model_dir)
    loaded_models: Dict[str, torch.nn.Module] = {}

    for model_name, model_class in MODEL_REGISTRY.items():
        checkpoint_path = model_dir / MODEL_FILENAMES[model_name]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing expert checkpoint: {checkpoint_path}")

        model = model_class(input_size=input_size).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        loaded_models[model_name] = model
        print(f"Loaded {model_name} model from {checkpoint_path}")

    return loaded_models


def run_prediction(
    csv_path: str | Path,
    model_dir: str | Path = "outputs",
    output_csv: str | Path | None = None,
    max_rows: int | None = None,
    single_step: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    model_dir = Path(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stage_log(1, 4, "Load Preprocessing Artifacts")
    artifacts_path = model_dir / "preprocessing_artifacts.pkl"
    inference_kwargs = {"csv_path": csv_path, "max_rows": max_rows}

    if artifacts_path.exists():
        artifacts = load_preprocessing_artifacts(artifacts_path)
        inference_kwargs.update(
            {
                "window_size": artifacts["window_size"],
                "feature_scaler": artifacts["feature_scaler"],
                "feature_columns": artifacts["feature_columns"],
                "num_modes": artifacts.get("num_modes", 5),
                "lag_steps": artifacts.get("lag_steps", 12),
                "stats_window": artifacts.get("stats_window", 12),
            }
        )
        print(f"Loaded preprocessing artifacts from {artifacts_path}")
    else:
        print(f"Warning: {artifacts_path} not found. Using a fresh scaler from the input CSV.")

    stage_log(2, 4, "Preprocessing and EWT Feature Preparation")
    if single_step:
        inference = prepare_single_forecast_data(**inference_kwargs)
        print("Single-step mode: using the most recent 12 rows to forecast the next 5-minute wind speed.")
    else:
        try:
            inference = prepare_inference_data(**inference_kwargs)
        except ValueError as exc:
            if "rolling prediction windows" not in str(exc):
                raise
            inference = prepare_single_forecast_data(**inference_kwargs)
            single_step = True
            print("Detected a 12-row input window. Switching to single-step next-5-minute forecasting mode.")

    stage_log(3, 4, "Load Trained Expert Models and DRL Policy")
    expert_models = load_expert_models(model_dir=model_dir, input_size=len(inference.feature_columns), device=device)
    expert_predictions = {
        model_name: predict_model(model, inference.X, device=device)
        for model_name, model in expert_models.items()
    }
    prediction_matrix = np.column_stack([expert_predictions[name] for name in MODEL_NAMES])

    dqn_path = model_dir / "dqn_policy.pth"
    if dqn_path.exists():
        agent = DQNAgent(state_size=8, action_size=len(MODEL_NAMES), device=device)
        agent.load(dqn_path)
        print(f"Loaded DQN policy from {dqn_path}")

        if single_step:
            latest_meta = inference.metadata_df.iloc[0]
            env = WindSelectionEnv(inference.metadata_df, prediction_matrix)
            state = env.reset()
            action = agent.select_action(state, greedy=True)
            selected_actions = np.asarray([action], dtype=np.int64)
            selected_models = [MODEL_NAMES[action]]
            hybrid_predictions = np.asarray([prediction_matrix[0, action]], dtype=np.float32)
        else:
            env = WindSelectionEnv(inference.metadata_df, prediction_matrix)
            hybrid_outputs = evaluate_policy(agent, env)
            selected_actions = hybrid_outputs["actions"]
            selected_models = [MODEL_NAMES[action] for action in selected_actions]
            hybrid_predictions = hybrid_outputs["predictions"]
    else:
        print(f"Warning: {dqn_path} not found. Falling back to CNN-GRU predictions.")
        selected_actions = np.full(len(inference.metadata_df), 2, dtype=np.int64)
        selected_models = ["CNN-GRU"] * len(inference.metadata_df)
        hybrid_predictions = expert_predictions["CNN-GRU"]

    stage_log(4, 4, "Generate Final Forecast and Save Output")
    if single_step:
        results_df = inference.metadata_df[["input_window_start", "input_window_end", "forecast_timestamp", "last_known_wind_speed_40m"]].copy()
        for model_name in MODEL_NAMES:
            column_name = f"pred_{model_name.lower().replace('-', '_')}"
            results_df[column_name] = expert_predictions[model_name]
        results_df["selected_action"] = selected_actions
        results_df["selected_model"] = selected_models
        results_df["pred_drl_hybrid"] = hybrid_predictions

        output_csv = _resolve_output_path(
            output_csv,
            default_dir=model_dir / "forecasts",
            default_prefix="next_5min_forecast",
            overwrite=overwrite,
        )
        results_df.to_csv(output_csv, index=False)

        print(f"Saved 1 next-step forecast to {output_csv}")
        print(
            f"Forecast for {results_df.loc[0, 'forecast_timestamp']} -> "
            f"{results_df.loc[0, 'pred_drl_hybrid']:.4f} m/s using {results_df.loc[0, 'selected_model']}"
        )
        _print_row_wise_forecast(results_df.iloc[0])
        return results_df

    results_df = inference.metadata_df[["timestamp", "wind_speed_40m"]].copy()
    results_df = results_df.rename(columns={"wind_speed_40m": "actual_wind_speed_40m"})

    for model_name in MODEL_NAMES:
        column_name = f"pred_{model_name.lower().replace('-', '_')}"
        results_df[column_name] = expert_predictions[model_name]

    results_df["selected_action"] = selected_actions
    results_df["selected_model"] = selected_models
    results_df["pred_drl_hybrid"] = hybrid_predictions
    results_df["hybrid_under_prediction"] = results_df["pred_drl_hybrid"] < results_df["actual_wind_speed_40m"]

    output_csv = _resolve_output_path(
        output_csv,
        default_dir=model_dir / "predictions",
        default_prefix="new_data_predictions",
        overwrite=overwrite,
    )
    results_df.to_csv(output_csv, index=False)

    rmse = float(np.sqrt(np.mean((results_df["actual_wind_speed_40m"] - results_df["pred_drl_hybrid"]) ** 2)))
    mae = float(np.mean(np.abs(results_df["actual_wind_speed_40m"] - results_df["pred_drl_hybrid"])))
    safety_violations = int(results_df["hybrid_under_prediction"].sum())
    plot_path = output_csv.with_name("actual_vs_predicted.png")
    plot_actual_vs_predicted(results_df, plot_path)

    print(f"Saved {len(results_df)} predictions to {output_csv}")
    print(f"Prediction check -> RMSE={rmse:.4f}, MAE={mae:.4f}, Safety_Violations={safety_violations}")
    print(f"Saved comparison plot to {plot_path}")
    print(results_df.head(10).to_string(index=False))
    return results_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load saved expert models and DQN policy to predict on a new CSV")
    parser.add_argument("--csv", required=True, help="Path to the new wind CSV file")
    parser.add_argument("--model-dir", default="outputs", help="Directory containing the .pth checkpoints")
    parser.add_argument("--output-csv", default=None, help="Where to save the prediction results CSV")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick testing")
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Use the most recent 12 rows to predict only the next 5-minute wind speed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite `--output-csv` if it exists. By default, a unique timestamped filename is used.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_prediction(
        csv_path=args.csv,
        model_dir=args.model_dir,
        output_csv=args.output_csv,
        max_rows=args.max_rows,
        single_step=args.single_step,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
