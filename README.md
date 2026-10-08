# Tremor Classification API

A machine learning-based API for classifying tremor types using accelerometer data from various motor tasks.

## Overview

This API uses a trained machine learning model to classify tremors based on accelerometer sensor data (x, y, z axes) collected during different motor tasks. The model extracts frequency-domain and time-domain features to predict the tremor type.

## Requirements

```bash
pip install fastapi pydantic numpy joblib scipy uvicorn
```

## Installation

1. Ensure you have the required model files in the project directory:
   - `tremor_model.pkl` - Trained ML model
   - `label_encoder.pkl` - Label encoder for tremor classes

2. Install dependencies:
```bash
pip install -r requirements.txt
```



### Required Input Format

The API expects accelerometer data from the following 5 tasks:
- `rest` - Patient at rest
- `finger_nose` - Finger-to-nose task
- `tapping` - Finger tapping task
- `eyes_closed` - Standing with eyes closed
- `initial` - Initial assessment

Each task must include:
- `sampling_rate` (int) - Sensor sampling rate in Hz
- `x` (list of floats) - X-axis accelerometer readings
- `y` (list of floats) - Y-axis accelerometer readings
- `z` (list of floats) - Z-axis accelerometer readings

### Sample Input

```json
{
  "rest": {
    "sampling_rate": 100,
    "x": [0.12, 0.15, 0.11, 0.13, 0.14, 0.12, 0.16, 0.13, 0.11, 0.15],
    "y": [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.07, 0.08, 0.10, 0.09],
    "z": [0.98, 0.97, 0.99, 0.98, 0.97, 0.98, 0.99, 0.98, 0.97, 0.98]
  },
  "finger_nose": {
    "sampling_rate": 100,
    "x": [0.45, 0.52, 0.48, 0.55, 0.50, 0.47, 0.53, 0.49, 0.51, 0.54],
    "y": [0.32, 0.35, 0.30, 0.37, 0.33, 0.31, 0.36, 0.34, 0.32, 0.35],
    "z": [0.85, 0.87, 0.84, 0.88, 0.86, 0.85, 0.87, 0.86, 0.84, 0.88]
  },
  "tapping": {
    "sampling_rate": 100,
    "x": [0.20, 0.25, 0.22, 0.27, 0.23, 0.21, 0.26, 0.24, 0.22, 0.25],
    "y": [0.15, 0.18, 0.16, 0.19, 0.17, 0.15, 0.18, 0.16, 0.17, 0.18],
    "z": [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.96, 0.95, 0.94, 0.96]
  },
  "eyes_closed": {
    "sampling_rate": 100,
    "x": [0.10, 0.12, 0.09, 0.13, 0.11, 0.10, 0.12, 0.11, 0.09, 0.13],
    "y": [0.08, 0.10, 0.07, 0.11, 0.09, 0.08, 0.10, 0.09, 0.08, 0.10],
    "z": [0.99, 0.98, 1.00, 0.97, 0.99, 0.98, 1.00, 0.99, 0.98, 0.99]
  },
  "initial": {
    "sampling_rate": 100,
    "x": [0.14, 0.17, 0.13, 0.18, 0.15, 0.14, 0.17, 0.16, 0.13, 0.18],
    "y": [0.10, 0.12, 0.09, 0.13, 0.11, 0.10, 0.12, 0.11, 0.10, 0.12],
    "z": [0.97, 0.96, 0.98, 0.95, 0.97, 0.96, 0.98, 0.97, 0.96, 0.97]
  }
}
```
## Feature Extraction

The model extracts the following features from each task:

1. **RMS (Root Mean Square)** - Measure of signal magnitude/intensity
2. **Dominant Frequency** - Peak frequency in the 3-12 Hz tremor band
3. **Band Ratio** - Ratio of power in 4-6 Hz vs 6-12 Hz bands
4. **Rest Movement Ratio** - Cross-task feature comparing rest vs. active movement

## Example Usage with Python

```python
import requests

url = "http://localhost:8000/predict"

data = {
    "rest": {
        "sampling_rate": 100,
        "x": [0.12, 0.15, 0.11, ...],
        "y": [0.08, 0.09, 0.07, ...],
        "z": [0.98, 0.97, 0.99, ...]
    },
    # ... other tasks
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Tremor Type: {result['predicted_class']}")
print(f"Confidence: {max(result['probabilities'].values()):.2%}")
```

## Example Usage with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

## Notes

- Accelerometer data should be in g-force units (standard accelerometer output)
- Sampling rate should be at least 50 Hz for accurate frequency analysis
- Minimum data duration: ~2-5 seconds per task recommended
- The model focuses on the 3-12 Hz frequency band, typical for tremor analysis
