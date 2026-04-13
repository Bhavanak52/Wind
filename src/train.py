from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.drl_agent import evaluate_policy, train_dqn_agent
from src.environment import WindSelectionEnv
from src.models import MODEL_REGISTRY
from src.preprocessing import PreparedData, prepare_datasets, save_preprocessing_artifacts
from src.visualize import plot_decomposition, plot_decision_timeline, save_metrics_table


class WindTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def stage_log(stage: int, title: str) -> None:
    print(f"\n[Stage {stage}/7] {title}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def create_loaders(prepared: PreparedData, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(WindTensorDataset(prepared.X_train, prepared.y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindTensorDataset(prepared.X_val, prepared.y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: str | Path,
    model_name: str,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                val_loss = criterion(predictions, batch_y)
                val_losses.append(val_loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"[{model_name}] Epoch {epoch:03d}/{epochs} | train_loss={mean_train:.5f} | val_loss={mean_val:.5f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    tensor_X = torch.as_tensor(X, dtype=torch.float32)

    for start in range(0, len(tensor_X), batch_size):
        batch = tensor_X[start : start + batch_size].to(device)
        preds = model(batch).detach().cpu().numpy()
        outputs.append(preds)

    return np.concatenate(outputs).astype(np.float32)


def train_base_models(
    prepared: PreparedData,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, Dict[str, np.ndarray | nn.Module | str]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    train_loader, val_loader = create_loaders(prepared, batch_size=batch_size)

    results: Dict[str, Dict[str, np.ndarray | nn.Module | str]] = {}
    for model_name, model_class in MODEL_REGISTRY.items():
        print(f"\nTraining expert model: {model_name}")
        model = model_class(input_size=prepared.X_train.shape[-1])
        checkpoint_path = output_dir / f"{model_name.replace('-', '_')}_model.pth"
        model = train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            save_path=checkpoint_path,
            model_name=model_name,
        )

        results[model_name] = {
            "model": model,
            "path": str(checkpoint_path),
            "val_predictions": predict_model(model, prepared.X_val, device=device),
            "test_predictions": predict_model(model, prepared.X_test, device=device),
        }

    return results


def evaluate_hybrid_selector(
    prepared: PreparedData,
    expert_results: Dict[str, Dict[str, np.ndarray | nn.Module | str]],
    dqn_episodes: int,
    output_dir: str | Path,
    device: Optional[torch.device] = None,
):
    output_dir = Path(output_dir)
    model_names = list(MODEL_REGISTRY.keys())
    val_predictions = np.column_stack([expert_results[name]["val_predictions"] for name in model_names])
    test_predictions = np.column_stack([expert_results[name]["test_predictions"] for name in model_names])

    print(f"\nTraining DQN master selector on {len(prepared.val_df)} validation samples...")
    val_env = WindSelectionEnv(prepared.val_df, val_predictions)
    agent, reward_history = train_dqn_agent(
        env=val_env,
        episodes=dqn_episodes,
        device=device,
        save_path=output_dir / "dqn_policy.pth",
        verbose_every=max(1, min(10, dqn_episodes)),
    )

    test_env = WindSelectionEnv(prepared.test_df, test_predictions)
    hybrid_outputs = evaluate_policy(agent, test_env)

    rewards_df = pd.DataFrame({
        "episode": np.arange(1, len(reward_history) + 1),
        "total_reward": reward_history,
    })
    rewards_df.to_csv(output_dir / "dqn_training_rewards.csv", index=False)

    decisions_df = prepared.test_df[["timestamp"]].copy()
    decisions_df["chosen_action"] = hybrid_outputs["actions"]
    decisions_df["chosen_model"] = [model_names[idx] for idx in hybrid_outputs["actions"]]
    decisions_df["hybrid_prediction"] = hybrid_outputs["predictions"]
    decisions_df["target"] = prepared.y_test
    decisions_df.to_csv(output_dir / "decision_trace.csv", index=False)

    return hybrid_outputs


def compute_metrics(prepared: PreparedData, expert_results, hybrid_outputs) -> pd.DataFrame:
    metrics = []

    for model_name, payload in expert_results.items():
        test_predictions = np.asarray(payload["test_predictions"], dtype=np.float32)
        metrics.append(
            {
                "Model": model_name,
                "MAE": round(mae(prepared.y_test, test_predictions), 4),
                "RMSE": round(rmse(prepared.y_test, test_predictions), 4),
                "MSE": round(mse(prepared.y_test, test_predictions), 4),
                "MAPE(%)": round(mape(prepared.y_test, test_predictions), 4),
                "R2": round(float(r2_score(prepared.y_test, test_predictions)), 4),
                "R": round(corrcoef(prepared.y_test, test_predictions), 4),
                "Safety_Violations": int(np.sum(test_predictions < prepared.y_test)),
            }
        )

    hybrid_predictions = np.asarray(hybrid_outputs["predictions"], dtype=np.float32)
    metrics.append(
        {
            "Model": "DRL-Hybrid",
            "MAE": round(mae(prepared.y_test, hybrid_predictions), 4),
            "RMSE": round(rmse(prepared.y_test, hybrid_predictions), 4),
            "MSE": round(mse(prepared.y_test, hybrid_predictions), 4),
            "MAPE(%)": round(mape(prepared.y_test, hybrid_predictions), 4),
            "R2": round(float(r2_score(prepared.y_test, hybrid_predictions)), 4),
            "R": round(corrcoef(prepared.y_test, hybrid_predictions), 4),
            "Safety_Violations": int(np.sum(hybrid_predictions < prepared.y_test)),
        }
    )

    return pd.DataFrame(metrics)


def run_pipeline(
    csv_path: str | Path,
    output_dir: str | Path = "outputs",
    window_size: int = 12,
    num_modes: int = 5,
    lag_steps: int = 12,
    stats_window: int = 12,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    epochs: int = 50,
    dqn_episodes: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    max_rows: Optional[int] = None,
):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stage_log(1, "Data Collection and Preprocessing")
    prepared = prepare_datasets(
        csv_path=csv_path,
        window_size=window_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_modes=num_modes,
        lag_steps=lag_steps,
        stats_window=stats_window,
        max_rows=max_rows,
    )
    print(
        f"Prepared data -> train: {prepared.X_train.shape}, val: {prepared.X_val.shape}, test: {prepared.X_test.shape}"
    )

    stage_log(2, "EWT Decomposition Artifacts")
    artifacts_path = save_preprocessing_artifacts(prepared, output_dir / "preprocessing_artifacts.pkl")
    print(f"Saved preprocessing artifacts to: {artifacts_path}")

    stage_log(3, "Decomposition Visualization")
    plot_decomposition(prepared.raw_df, output_dir / "decomposition.png")

    stage_log(4, "Train Individual Deep Models")
    expert_results = train_base_models(
        prepared=prepared,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        output_dir=output_dir,
    )

    stage_log(5, "Train DRL Agent for Dynamic Selection")
    hybrid_outputs = evaluate_hybrid_selector(
        prepared=prepared,
        expert_results=expert_results,
        dqn_episodes=dqn_episodes,
        output_dir=output_dir,
        device=device,
    )

    stage_log(6, "Evaluation Metrics and Comparison")
    metrics_df = compute_metrics(prepared, expert_results, hybrid_outputs)
    save_metrics_table(metrics_df.to_dict(orient="records"), output_dir / "metrics_comparison.csv")
    plot_decision_timeline(prepared.test_df["timestamp"], hybrid_outputs["actions"], output_dir / "decision_timeline.png")

    stage_log(7, "Final Test Predictions Export")
    prediction_frame = prepared.test_df[["timestamp"]].copy()
    prediction_frame["target"] = prepared.y_test
    for model_name, payload in expert_results.items():
        prediction_frame[f"pred_{model_name.lower().replace('-', '_')}"] = payload["test_predictions"]
    prediction_frame["pred_drl_hybrid"] = hybrid_outputs["predictions"]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))
    return metrics_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid wind speed forecasting with expert models + DQN selector")
    parser.add_argument("--csv", default="wind_data.csv", help="Path to wind_data.csv")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved models and figures")
    parser.add_argument("--window-size", type=int, default=12, help="Number of 5-minute steps per input window")
    parser.add_argument("--num-modes", type=int, default=5, help="Number of EWT-style decomposition modes (typical: 4-8)")
    parser.add_argument("--lag-steps", type=int, default=12, help="Number of lagged target steps to fuse as features")
    parser.add_argument("--stats-window", type=int, default=12, help="Rolling window for statistical mode features")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for each expert model")
    parser.add_argument("--dqn-episodes", type=int, default=200, help="Episodes for DQN master selector")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit for quick experiments")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(
        csv_path=args.csv,
        output_dir=args.output_dir,
        window_size=args.window_size,
        num_modes=args.num_modes,
        lag_steps=args.lag_steps,
        stats_window=args.stats_window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        dqn_episodes=args.dqn_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
