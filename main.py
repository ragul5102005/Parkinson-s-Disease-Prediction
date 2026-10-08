from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import joblib
from scipy import signal, fft
import json
import asyncio
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

app = FastAPI(title="Tremor Classification API", version="1.0.0")

# Canonical class mapping for model outputs.
LABEL_MAP = {
    1: "HEALTHY",
    2: "PARKINSON",
}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    # Prefer newer magnitude-only models; keep legacy fallback for compatibility.
    model_file_candidates = {
        "task0": ["task0_magnitude_model.pkl", "task0_model.pkl"],
        "task1": ["task1_magnitude_model.pkl", "task1_model.pkl"],
        "task2": ["task2_magnitude_model.pkl", "task2_model.pkl"],
        "task3": ["task3_magnitude_model.pkl", "task3_model.pkl"],
    }

    models = {}
    for model_name, candidates in model_file_candidates.items():
        loaded = False
        for file_name in candidates:
            try:
                models[model_name] = joblib.load(file_name)
                print(f"Loaded {model_name} from {file_name}")
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            raise FileNotFoundError(
                f"Could not load any model file for {model_name}: {candidates}"
            )

    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = {}

# Data Models
class TaskData(BaseModel):
    sampling_rate: int
    x: List[float]
    y: List[float]
    z: List[float]

class TremorInput(BaseModel):
    rest: TaskData
    finger_nose: TaskData
    tapping: TaskData
    eyes_closed: TaskData

class ArduinoStreamData(BaseModel):
    task: str  # rest, finger_nose, tapping, eyes_closed, initial
    x: float
    y: float
    z: float
    sampling_rate: int = 100

# Global storage for streaming data
streaming_data = defaultdict(lambda: {"x": [], "y": [], "z": [], "sampling_rate": 100})


# Feature Extraction Functions
def calculate_rms(data: np.ndarray) -> float:
    """Calculate Root Mean Square of the signal."""
    return np.sqrt(np.mean(data ** 2))

def find_dominant_frequency(data: np.ndarray, sampling_rate: int, freq_range=(3, 12)) -> float:
    """Find dominant frequency in the tremor band (3-12 Hz)."""
    # Perform FFT
    n = len(data)
    fft_vals = fft.fft(data)
    fft_freq = fft.fftfreq(n, 1/sampling_rate)
    
    # Get positive frequencies only
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    
    # Filter to tremor frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    tremor_freqs = freqs[freq_mask]
    tremor_power = power[freq_mask]
    
    if len(tremor_power) == 0:
        return 0.0
    
    # Find peak frequency
    peak_idx = np.argmax(tremor_power)
    return float(tremor_freqs[peak_idx])

def calculate_band_ratio(data: np.ndarray, sampling_rate: int) -> float:
    """Calculate ratio of power in 4-6 Hz band vs 6-12 Hz band."""
    n = len(data)
    fft_vals = fft.fft(data)
    fft_freq = fft.fftfreq(n, 1/sampling_rate)
    
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    
    # Calculate power in each band
    band1_mask = (freqs >= 4) & (freqs <= 6)
    band2_mask = (freqs >= 6) & (freqs <= 12)
    
    band1_power = np.sum(power[band1_mask])
    band2_power = np.sum(power[band2_mask])
    
    if band2_power == 0:
        return 0.0
    
    return float(band1_power / band2_power)

def extract_features_from_task(task_data: TaskData) -> Dict[str, float]:
    """Extract features from a single task's accelerometer data."""
    x = np.array(task_data.x)
    y = np.array(task_data.y)
    z = np.array(task_data.z)
    
    # Calculate magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    features = {
        "rms_x": calculate_rms(x),
        "rms_y": calculate_rms(y),
        "rms_z": calculate_rms(z),
        "rms_magnitude": calculate_rms(magnitude),
        "dominant_freq_x": find_dominant_frequency(x, task_data.sampling_rate),
        "dominant_freq_y": find_dominant_frequency(y, task_data.sampling_rate),
        "dominant_freq_z": find_dominant_frequency(z, task_data.sampling_rate),
        "band_ratio_x": calculate_band_ratio(x, task_data.sampling_rate),
        "band_ratio_y": calculate_band_ratio(y, task_data.sampling_rate),
        "band_ratio_z": calculate_band_ratio(z, task_data.sampling_rate),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_z": float(np.mean(z)),
        "std_x": float(np.std(x)),
        "std_y": float(np.std(y)),
        "std_z": float(np.std(z)),
    }
    
    return features

def calculate_rest_movement_ratio(rest_rms: float, active_rms: float) -> float:
    """Calculate ratio of rest movement to active movement."""
    if active_rms == 0:
        return 0.0
    return rest_rms / active_rms

def _softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw decision scores to normalized probabilities."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Convert binary decision scores to probabilities."""
    return 1.0 / (1.0 + np.exp(-scores))


def decode_label(label_value) -> str:
    """Convert model class ids to human-readable labels."""
    try:
        numeric_value = int(label_value)
        return LABEL_MAP.get(numeric_value, str(label_value))
    except (TypeError, ValueError):
        return str(label_value)


def extract_task_features(task_data: TaskData) -> np.ndarray:
    """Convert each x/y/z sample into its magnitude for model input."""
    x = np.asarray(task_data.x, dtype=float)
    y = np.asarray(task_data.y, dtype=float)
    z = np.asarray(task_data.z, dtype=float)

    if not (len(x) == len(y) == len(z)):
        raise ValueError("Task input arrays x, y, and z must have the same length")
    if len(x) == 0:
        raise ValueError("Task input contains no samples")

    magnitudes = np.sqrt(x**2 + y**2 + z**2)
    if pd is not None:
        feature_vector = pd.DataFrame({"magnitude": magnitudes})
    else:
        feature_vector = magnitudes.reshape(-1, 1)

    return feature_vector, {
        "average_x": float(np.mean(x)),
        "average_y": float(np.mean(y)),
        "average_z": float(np.mean(z)),
        "average_magnitude": float(np.mean(magnitudes)),
        "min_magnitude": float(np.min(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)),
        "samples": int(len(magnitudes)),
    }


def predict_with_confidence(model, features: np.ndarray) -> Dict[str, object]:
    """Run prediction and produce a confidence distribution for different model types."""
    classes = getattr(model, "classes_", None)
    sample_predictions = model.predict(features)

    if classes is None or len(classes) == 0:
        unique_classes = np.unique(sample_predictions)
        classes = unique_classes

    if hasattr(model, "predict_proba"):
        sample_probabilities = model.predict_proba(features)
        probabilities = np.mean(sample_probabilities, axis=0)
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(features)
        if np.ndim(decision_scores) == 1:
            positive_class_probabilities = _sigmoid(np.asarray(decision_scores, dtype=float))
            sample_probabilities = np.column_stack([
                1.0 - positive_class_probabilities,
                positive_class_probabilities,
            ])
        else:
            sample_probabilities = _softmax(np.asarray(decision_scores, dtype=float))
        probabilities = np.mean(sample_probabilities, axis=0)
    else:
        vote_counts = np.array([
            np.sum(sample_predictions == class_name) for class_name in classes
        ], dtype=float)
        probabilities = vote_counts / vote_counts.sum()

    best_index = int(np.argmax(probabilities))
    final_class = classes[best_index]

    return {
        "prediction": decode_label(final_class),
        "prediction_raw": str(final_class),
        "probabilities": probabilities.tolist(),
        "confidence": float(probabilities[best_index]),
        "sample_predictions": [str(prediction) for prediction in sample_predictions.tolist()],
    }


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Tremor Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict tremor type from complete task data",
            "/stream": "WebSocket - Stream Arduino data in real-time",
            "/process-stream": "POST - Process accumulated stream data",
            "/clear-stream": "POST - Clear accumulated stream data",
            "/docs": "API documentation"
        }
    }

@app.post("/predict")
async def predict_tremor(data: TremorInput):
    """
    Predict tremor type from accelerometer data.
    
    Expects data from 4 tasks: rest, finger_nose, tapping, eyes_closed.
    Each task is mapped to its corresponding model.
    """
    try:
        # Map tasks to their corresponding models
        task_model_mapping = [
            ("rest", data.rest, "task0"),
            ("finger_nose", data.finger_nose, "task1"),
            ("tapping", data.tapping, "task2"),
            ("eyes_closed", data.eyes_closed, "task3")
        ]
        
        # Get predictions from each model using its corresponding task data
        predictions = {}
        for task_name, task_data, model_name in task_model_mapping:
            if model_name in models:
                features, averages = extract_task_features(task_data)
                prediction_result = predict_with_confidence(models[model_name], features)
                predictions[model_name] = {
                    "prediction": prediction_result["prediction"],
                    "probabilities": prediction_result["probabilities"],
                    "confidence": prediction_result["confidence"],
                    "task": task_name,
                    "averages": averages
                }
        
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/stream-data")
async def receive_stream_data(data: ArduinoStreamData):
    """
    Receive streaming data from Arduino.
    Data is accumulated for each task until ready for processing.
    """
    try:
        task = data.task
        
        # Print when a new task starts
        if len(streaming_data[task]["x"]) == 0:
            print(f"\n🎯 Starting task: {task.upper()}")
        
        streaming_data[task]["x"].append(data.x)
        streaming_data[task]["y"].append(data.y)
        streaming_data[task]["z"].append(data.z)
        streaming_data[task]["sampling_rate"] = data.sampling_rate
        
        samples = len(streaming_data[task]["x"])
        
        # Print every 50 samples to show progress
        if samples % 50 == 0:
            print(f"📊 {task.upper()}: {samples} samples collected")
        
        return {
            "status": "success",
            "task": task,
            "samples_collected": samples,
            "message": f"Data point added to {task} task"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing stream data: {str(e)}")

@app.post("/process-stream")
async def process_stream():
    """
    Process all accumulated streaming data and make prediction.
    """
    try:
        # Check if all required tasks have data
        required_tasks = ["rest", "finger_nose", "tapping", "eyes_closed"]
        missing_tasks = [task for task in required_tasks if len(streaming_data[task]["x"]) == 0]
        
        if missing_tasks:
            return {
                "status": "incomplete",
                "message": f"Missing data for tasks: {', '.join(missing_tasks)}",
                "collected_tasks": [task for task in required_tasks if len(streaming_data[task]["x"]) > 0]
            }
        
        # Build TremorInput from streaming data
        tremor_input = TremorInput(
            rest=TaskData(
                sampling_rate=streaming_data["rest"]["sampling_rate"],
                x=streaming_data["rest"]["x"],
                y=streaming_data["rest"]["y"],
                z=streaming_data["rest"]["z"]
            ),
            finger_nose=TaskData(
                sampling_rate=streaming_data["finger_nose"]["sampling_rate"],
                x=streaming_data["finger_nose"]["x"],
                y=streaming_data["finger_nose"]["y"],
                z=streaming_data["finger_nose"]["z"]
            ),
            tapping=TaskData(
                sampling_rate=streaming_data["tapping"]["sampling_rate"],
                x=streaming_data["tapping"]["x"],
                y=streaming_data["tapping"]["y"],
                z=streaming_data["tapping"]["z"]
            ),
            eyes_closed=TaskData(
                sampling_rate=streaming_data["eyes_closed"]["sampling_rate"],
                x=streaming_data["eyes_closed"]["x"],
                y=streaming_data["eyes_closed"]["y"],
                z=streaming_data["eyes_closed"]["z"]
            )
        )
        
        # Print task completion summary
        print("\n" + "="*50)
        print("✅ ALL TASKS COMPLETED!")
        print("="*50)
        for task in required_tasks:
            count = len(streaming_data[task]["x"])
            print(f"  {task.upper():15} : {count:4} samples")
        print("="*50)
        
        # Check if models are loaded
        if not models:
            error_msg = "Models not loaded. Install dependencies: pip install imbalanced-learn scikit-learn"
            print(f"\n❌ ERROR: {error_msg}\n")
            return {
                "status": "error",
                "error": error_msg,
                "predictions": {},
                "message": "Models failed to load. Check backend console for details."
            }
        
        # Map tasks to their corresponding models
        task_model_mapping = [
            ("rest", tremor_input.rest, "task0"),
            ("finger_nose", tremor_input.finger_nose, "task1"),
            ("tapping", tremor_input.tapping, "task2"),
            ("eyes_closed", tremor_input.eyes_closed, "task3")
        ]
        
        print(f"\n📐 Processing each task with its model:")
        print("="*50)
        
        predictions = {}
        task_averages = {}
        for task_name, task_data, model_name in task_model_mapping:
            try:
                # Extract features for this specific task
                features, averages = extract_task_features(task_data)
                task_averages[task_name] = averages
                
                print(f"\n{task_name.upper()} → {model_name}")
                print(f"  Avg X: {averages['average_x']:.4f}")
                print(f"  Avg Y: {averages['average_y']:.4f}")
                print(f"  Avg Z: {averages['average_z']:.4f}")
                print(f"  Avg Magnitude: {averages['average_magnitude']:.4f}")
                print(f"  Magnitude Samples: {averages['samples']}")
                
                # Predict with corresponding model
                if model_name in models:
                    prediction_result = predict_with_confidence(models[model_name], features)
                    predictions[model_name] = {
                        "prediction": prediction_result["prediction"],
                        "probabilities": prediction_result["probabilities"],
                        "confidence": prediction_result["confidence"],
                        "task": task_name
                    }
                    max_prob = prediction_result["confidence"] * 100
                    print(f"  → {prediction_result['prediction']} ({max_prob:.1f}%)")
                else:
                    predictions[model_name] = {"error": f"Model {model_name} not loaded"}
                    print(f"  → ERROR: Model not loaded")
            except Exception as e:
                predictions[model_name] = {"error": str(e)}
                print(f"  → ERROR: {str(e)}")
        
        print("="*50 + "\n")
        
        # Get sample counts for each task
        sample_counts = {task: len(streaming_data[task]["x"]) for task in required_tasks}
        
        return {
            "status": "success",
            "averages": task_averages,
            "predictions": predictions,
            "sample_counts": sample_counts,
            "message": "Stream data processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing stream: {str(e)}")

@app.post("/clear-stream")
async def clear_stream(task: str = None):
    """
    Clear accumulated streaming data.
    If task is specified, clears only that task. Otherwise clears all tasks.
    """
    try:
        if task:
            streaming_data[task] = {"x": [], "y": [], "z": [], "sampling_rate": 100}
            return {"status": "success", "message": f"Cleared data for task: {task}"}
        else:
            streaming_data.clear()
            return {"status": "success", "message": "Cleared all streaming data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing stream: {str(e)}")

@app.get("/stream-status")
async def get_stream_status():
    """Get the current status of streaming data collection."""
    status = {}
    for task in ["rest", "finger_nose", "tapping", "eyes_closed"]:
        status[task] = {
            "samples_collected": len(streaming_data[task]["x"]),
            "has_data": len(streaming_data[task]["x"]) > 0
        }
    
    return {
        "status": "success",
        "tasks": status,
        "total_samples": sum(len(streaming_data[task]["x"]) for task in status.keys())
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Arduino data streaming.
    """
    await websocket.accept()
    try:
        while True:
            # Receive data from Arduino
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            # Process the incoming data
            task = data_json.get("task")
            x = data_json.get("x")
            y = data_json.get("y")
            z = data_json.get("z")
            sampling_rate = data_json.get("sampling_rate", 100)
            
            # Store data
            streaming_data[task]["x"].append(x)
            streaming_data[task]["y"].append(y)
            streaming_data[task]["z"].append(z)
            streaming_data[task]["sampling_rate"] = sampling_rate
            
            # Send acknowledgment
            await websocket.send_json({
                "status": "received",
                "task": task,
                "samples": len(streaming_data[task]["x"])
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
