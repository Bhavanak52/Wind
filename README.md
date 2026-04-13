<<<<<<< HEAD
# Wind Speed Forecasting Hybrid System

This project builds a **5-minute ahead wind speed forecaster** for the 40 m level using:

- **EMD/EWT-style decomposition** into trend + oscillation + noise
- **Three expert PyTorch models**: `LSTM`, `CNN-LSTM`, `CNN-GRU`
- **A DQN master selector** that chooses the safest expert at every time step
- **Safety-aware reward shaping** with a heavier penalty for under-prediction

## Project structure

```text
wind_speed_forecasting/
├── wind_data.csv
├── main.py
├── requirements.txt
├── outputs/
└── src/
    ├── preprocessing.py
    ├── models.py
    ├── environment.py
    ├── drl_agent.py
    ├── train.py
    └── visualize.py
```

## Run

```powershell
C:/Users/Bhavana/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install -r requirements.txt
C:/Users/Bhavana/AppData/Local/Python/pythoncore-3.14-64/python.exe main.py --epochs 50 --dqn-episodes 200
```

## Quick smoke test

```powershell
C:/Users/Bhavana/AppData/Local/Python/pythoncore-3.14-64/python.exe main.py --epochs 1 --dqn-episodes 2 --max-rows 1000
```

## Predict on a new CSV

After training, you can load the saved models in `outputs/` and predict on another file with the same schema:

```powershell
C:/Users/Bhavana/AppData/Local/Python/pythoncore-3.14-64/python.exe predict.py --csv wind_data.csv --model-dir outputs --output-csv outputs/new_data_predictions.csv
```

This automatically loads:

- `outputs/LSTM_model.pth`
- `outputs/CNN_LSTM_model.pth`
- `outputs/CNN_GRU_model.pth`
- `outputs/dqn_policy.pth`
- `outputs/preprocessing_artifacts.pkl` (if available)

## Expected outputs

The pipeline writes these artifacts to `outputs/`:

- `decomposition.png`
- `decision_timeline.png`
- `metrics_comparison.csv`
- `LSTM_model.pth`
- `CNN_LSTM_model.pth`
- `CNN_GRU_model.pth`
- `dqn_policy.pth`
- `preprocessing_artifacts.pkl`
- `decision_trace.csv`
- `test_predictions.csv`
- `new_data_predictions.csv`
=======
# Wind
>>>>>>> eb69b85a27cab1e766fea3b9c2f16ad9150429ec
