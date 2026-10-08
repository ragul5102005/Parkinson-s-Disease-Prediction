"""
Test script to verify Arduino-Backend communication
This simulates Arduino sending data to the FastAPI backend
"""
import requests
import time
import random

BACKEND_URL = "http://localhost:8000"

def test_stream_data():
    """Test streaming data to backend"""
    print("=" * 50)
    print("Testing Arduino Data Stream")
    print("=" * 50)
    
    tasks = ["rest", "finger_nose", "tapping", "eyes_closed", "initial"]
    
    # Clear any existing data
    print("\n[1] Clearing backend data...")
    response = requests.post(f"{BACKEND_URL}/clear-stream")
    print(f"    Status: {response.status_code}")
    
    # Send sample data for each task
    print("\n[2] Sending sample data for each task...")
    for task in tasks:
        print(f"\n    Task: {task}")
        for i in range(100):  # Send 100 samples per task
            data = {
                "task": task,
                "x": random.uniform(-0.5, 0.5),
                "y": random.uniform(-0.5, 0.5),
                "z": random.uniform(0.8, 1.2),
                "sampling_rate": 50
            }
            response = requests.post(f"{BACKEND_URL}/stream-data", json=data)
            if i % 20 == 0:
                print(f"      Sent {i+1} samples...")
        print(f"      ✓ Completed ({response.json()['samples_collected']} samples)")
    
    # Check status
    print("\n[3] Checking stream status...")
    response = requests.get(f"{BACKEND_URL}/stream-status")
    status = response.json()
    print(f"    Total samples: {status['total_samples']}")
    for task, info in status['tasks'].items():
        print(f"      {task}: {info['samples_collected']} samples")
    
    # Process and get prediction
    print("\n[4] Processing prediction...")
    response = requests.post(f"{BACKEND_URL}/process-stream")
    result = response.json()
    
    if result['status'] == 'success':
        print(f"    ✓ Success!")
        print(f"\n    Averages:")
        for task_name, averages in result['averages'].items():
            print(f"      {task_name}:")
            print(f"        X: {averages['average_x']:.4f}")
            print(f"        Y: {averages['average_y']:.4f}")
            print(f"        Z: {averages['average_z']:.4f}")
            print(f"        Magnitude: {averages['average_magnitude']:.4f}")
        
        print(f"\n    Predictions from models:")
        for model_name, pred_data in result['predictions'].items():
            if 'prediction' in pred_data:
                print(f"      {model_name}: {pred_data['prediction']}")
                if 'confidence' in pred_data:
                    max_prob = pred_data['confidence'] * 100
                    print(f"        Confidence: {max_prob:.1f}%")
                elif 'probabilities' in pred_data:
                    max_prob = max(pred_data['probabilities']) * 100
                    print(f"        Confidence: {max_prob:.1f}%")
    else:
        print(f"    ✗ Error: {result.get('message', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        # Test if backend is running
        print("Checking if backend is running...")
        response = requests.get(f"{BACKEND_URL}/")
        print(f"✓ Backend is online!\n")
        
        test_stream_data()
        
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to backend!")
        print("  Make sure the FastAPI server is running:")
        print("    python main.py")
    except Exception as e:
        print(f"✗ ERROR: {e}")
