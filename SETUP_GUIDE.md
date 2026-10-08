# Setup Guide for Arduino Sequential Task System

## Overview
This system allows an ESP32 with accelerometer to collect tremor data across 5 tasks, send it to a FastAPI backend, and display the prediction result.

## Architecture

```
Arduino ESP32 (Sensor) → WiFi → FastAPI Backend (PC) → ML Models → Prediction Result
                                                      ↓
                                               Display on OLED
```

## Setup Steps

### 1. Backend Setup (PC)

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Find Your PC's IP Address
**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" under your WiFi adapter (e.g., `192.168.1.100`)

**Linux/Mac:**
```bash
ifconfig
```
or
```bash
ip addr show
```

#### Update Backend URL in Arduino Code
Open `arduino_sequential_tasks.ino` and update line 30:
```cpp
const char* backend_url = "http://YOUR_PC_IP:8000";  // e.g., "http://192.168.1.100:8000"
```

#### Start the Backend Server
```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
Models loaded successfully!
```

#### Test Backend (Optional)
In a new terminal:
```bash
python test_arduino.py
```

### 2. Arduino Setup

#### Required Libraries
Install these in Arduino IDE (Tools → Manage Libraries):
1. **U8g2** by oliver
2. **ArduinoJson** by Benoit Blanchon (version 6.x)

#### WiFi Configuration
Update lines 27-28 in `arduino_sequential_tasks.ino`:
```cpp
const char* ssid     = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```

#### Upload Code
1. Connect ESP32 via USB
2. Select board: **ESP32 Dev Module**
3. Select correct COM port
4. Click Upload

### 3. Hardware Connections

#### OLED Display (SSD1309 2.42")
```
OLED Pin    →  ESP32 Pin
─────────────────────────
GND         →  GND
VCC         →  3.3V
SCK (CLK)   →  GPIO 18
SDA (MOSI)  →  GPIO 23
RES (RST)   →  GPIO 17
DC          →  GPIO 16
CS          →  GPIO 5
```

#### Accelerometer (Analog 3-axis)
```
Sensor Pin  →  ESP32 Pin
─────────────────────────
VCC         →  3.3V
GND         →  GND
X-OUT       →  GPIO 34 (ADC1_CH6)
Y-OUT       →  GPIO 35 (ADC1_CH7)
Z-OUT       →  GPIO 32 (ADC1_CH4)
```

## Usage

### Running the Test

1. **Start Backend** (on PC):
   ```bash
   python main.py
   ```

2. **Power on ESP32** and open Serial Monitor (115200 baud)

3. **Wait for WiFi Connection**
   - OLED will show IP address
   - Serial monitor shows "System Ready!"

4. **Sequential Task Execution:**

   The system will prompt you through 5 tasks:
   
   **Task 1: REST**
   - Patient sits still with hands on lap
   - Press ENTER to start
   - 5 seconds of data collection
   
   **Task 2: FINGER-NOSE**
   - Patient touches finger to nose repeatedly
   - Press ENTER to start
   - 5 seconds of data collection
   
   **Task 3: TAPPING**
   - Patient taps fingers together
   - Press ENTER to start
   - 5 seconds of data collection
   
   **Task 4: EYES CLOSED**
   - Patient stands with eyes closed
   - Press ENTER to start
   - 5 seconds of data collection
   
   **Task 5: INITIAL**
   - Initial assessment position
   - Press ENTER to start
   - 5 seconds of data collection

5. **View Results:**
   - After all tasks complete, system shows "Processing..."
   - Prediction appears on OLED and Serial Monitor
   - Shows predicted class and confidence percentage

### Serial Monitor Output Example

```
================================
 ESP32 PARKINSON DETECTION
 SEQUENTIAL TASK MODE
================================
Starting OLED... OK!
Connecting to WiFi.........
WiFi Connected!
IP: 192.168.1.150

================================
System Ready!
================================

Task 1: REST
Press ENTER to start...

*** STARTING Task 1 ***
Duration: 5 seconds
Collecting data...

Task 1 COMPLETED - Samples: 250

Task 2: FINGER-NOSE
Press ENTER to start...

[... continues for all tasks ...]

=================================
ALL TASKS COMPLETED!
Processing prediction...
=================================

=== FINAL RESULT ===
Prediction: HEALTHY
Confidence: 92.3%
====================
```

## Troubleshooting

### OLED Not Displaying
- Check wiring connections
- Verify power (3.3V, not 5V!)
- Try swapping DC and RST pins in code

### WiFi Connection Failed
- Double-check SSID and password
- Ensure ESP32 is within WiFi range
- Check if WiFi supports 2.4GHz (ESP32 doesn't support 5GHz)

### Backend Connection Error
- Verify PC IP address is correct in Arduino code
- Check firewall isn't blocking port 8000
- Ensure PC and ESP32 are on same network
- Test backend: open browser to `http://localhost:8000`

### Sensor Reading Issues
- Verify sensor is powered (3.3V)
- Use only ADC1 pins (ADC2 conflicts with WiFi)
- Check sensor ground connection

### Model Not Loading
- Ensure `.pkl` files are in same directory as `main.py`
- Check Python version compatibility (3.8+)
- Verify joblib/scikit-learn versions match

## Customization

### Adjust Task Duration
In `arduino_sequential_tasks.ino`, modify the `duration_ms` field:
```cpp
Task tasks[] = {
  {"Task 1", "REST", "rest", 10000, 0},  // 10 seconds instead of 5
  // ...
};
```

### Change Sampling Rate
Modify line 462 in Arduino code:
```cpp
vTaskDelay(pdMS_TO_TICKS(20)); // 50Hz = 20ms delay
```
For 100Hz: use `10` instead of `20`

### Adjust Sensor Calibration
Lines 455-457, change conversion factors based on your sensor specs:
```cpp
float x = (rawX - 2048) / 1024.0;  // Adjust offset and scale
```

## API Endpoints

- `GET /` - API information
- `POST /stream-data` - Receive single data point
- `POST /process-stream` - Process all data and get prediction  
- `GET /stream-status` - Check collection status
- `POST /clear-stream` - Clear accumulated data
- `POST /predict` - Direct prediction with complete JSON data

## Next Steps

1. Calibrate sensor with known reference (1G gravity)
2. Collect training data from real patients
3. Train custom models with your data
4. Adjust task durations based on medical protocols
5. Add data logging/export features
6. Implement multi-user support

## Support

For issues or questions:
1. Check Serial Monitor output for errors
2. Test backend with `test_arduino.py`
3. Verify all connections with multimeter
4. Check library versions
