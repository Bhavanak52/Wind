import pandas as pd

from predict import run_prediction


def main() -> None:
    data = pd.read_csv("wind_data.csv")
    windows = [(0, 12), (120, 132), (240, 252), (480, 492), (720, 732), (960, 972)]

    print("Checking selector behavior across different 12-row windows...")
    for start, end in windows:
        path = f"tmp_window_{start}_{end}.csv"
        data.iloc[start:end].to_csv(path, index=False)

        out = run_prediction(path, model_dir="outputs", single_step=True)
        row = out.iloc[0]
        print(
            f"SUMMARY window {start}-{end}: "
            f"forecast={row['forecast_timestamp']}, "
            f"model={row['selected_model']}, "
            f"action={int(row['selected_action'])}, "
            f"pred={float(row['pred_drl_hybrid']):.4f}"
        )


if __name__ == "__main__":
    main()
