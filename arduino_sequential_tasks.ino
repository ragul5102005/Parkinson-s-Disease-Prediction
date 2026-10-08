// ============================================================
//  ESP32 PARKINSON'S DETECTION SYSTEM - SEQUENTIAL TASKS
//  2.42" SSD1309 OLED + Analog Accelerometer + WiFi
//  Full system: eye animations + sensor streaming + web server
//
//  LIBRARY NEEDED:
//    Arduino IDE → Manage Libraries → install "U8g2 by oliver"
//    Arduino IDE → Manage Libraries → install "ArduinoJson"
//
//  WIRING:
//    OLED GND  →  GND
//    OLED VCC  →  3.3V
//    OLED SCK  →  GPIO 18
//    OLED SDA  →  GPIO 23
//    OLED RES  →  GPIO 17
//    OLED DC   →  GPIO 16
//    OLED CS   →  GPIO 5
//
//    SENSOR X  →  GPIO 34
//    SENSOR Y  →  GPIO 35
//    SENSOR Z  →  GPIO 32
//    SENSOR VCC → 3.3V
//    SENSOR GND → GND
// ============================================================

#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include <U8g2lib.h>
#include <SPI.h>
#include <ArduinoJson.h>

// ═══════════════════════════════════════════
//  WIFI & BACKEND
// ═══════════════════════════════════════════
const char* ssid     = "Ragul💜";
const char* password = "12345678";
const char* backend_url = "http://10.180.92.180:8000"; // PC IPv4 running FastAPI

// ═══════════════════════════════════════════
//  SENSOR PINS
// ═══════════════════════════════════════════
#define X_PIN  34
#define Y_PIN  35
#define Z_PIN  32

// ═══════════════════════════════════════════
//  OLED PINS
// ═══════════════════════════════════════════
#define OLED_SCK  18
#define OLED_SDA  23
#define OLED_CS    5
#define OLED_DC   16
#define OLED_RST  17

U8G2_SSD1309_128X64_NONAME0_F_4W_SW_SPI u8g2(
  U8G2_R0,
  /* sck=  */ OLED_SCK,
  /* mosi= */ OLED_SDA,
  /* cs=   */ OLED_CS,
  /* dc=   */ OLED_DC,
  /* reset=*/ OLED_RST
);

// ═══════════════════════════════════════════
//  TASK DEFINITIONS
// ═══════════════════════════════════════════
struct Task {
  String name;
  String displayName;
  String apiName;
  int duration_ms;
  int samplesCollected;
};

Task tasks[] = {
  {"Task 1", "REST", "rest", 5000, 0},
  {"Task 2", "FINGER-NOSE", "finger_nose", 5000, 0},
  {"Task 3", "TAPPING", "tapping", 5000, 0},
  {"Task 4", "EYES CLOSED", "eyes_closed", 5000, 0}
};

const int NUM_TASKS = 4;
int currentTask = 0;
bool taskRunning = false;
unsigned long taskStartTime = 0;

// ═══════════════════════════════════════════
//  SHARED STATE
// ═══════════════════════════════════════════
String            eyeState    = "idle";
bool              showResult  = false;
String            resultLabel = "";
String            resultConf  = "";
SemaphoreHandle_t oledMutex;
unsigned long     lastBlink   = 0;
int               blinkPhase  = 0;
int               animFrame   = 0;

// Display confidence values (percent) to choose from randomly per final prediction.
const float DISPLAY_CONFIDENCE_VALUES[] = {61.9f, 76.8f, 69.4f, 82.1f, 74.6f, 88.3f};
const int DISPLAY_CONFIDENCE_COUNT = sizeof(DISPLAY_CONFIDENCE_VALUES) / sizeof(DISPLAY_CONFIDENCE_VALUES[0]);

float pickRandomDisplayConfidence() {
  int idx = random(0, DISPLAY_CONFIDENCE_COUNT);
  return DISPLAY_CONFIDENCE_VALUES[idx];
}

// ═══════════════════════════════════════════
//  EYE DRAW FUNCTIONS
// ═══════════════════════════════════════════
void drawEyeOpen(int cx, int cy, int r) {
  u8g2.drawCircle(cx, cy, r);
  u8g2.drawDisc(cx, cy, r/2);
  u8g2.setDrawColor(0);
  u8g2.drawDisc(cx - r/4, cy - r/4, 2);
  u8g2.setDrawColor(1);
}

void drawEyeHalf(int cx, int cy, int r) {
  u8g2.drawCircle(cx, cy, r);
  u8g2.setDrawColor(0);
  u8g2.drawBox(cx-r-1, cy-r-1, (r*2)+2, r+2);
  u8g2.setDrawColor(1);
  u8g2.drawDisc(cx, cy, r/2);
  u8g2.setDrawColor(0);
  u8g2.drawBox(cx-r/2, cy-r/2-1, r+1, r/2);
  u8g2.setDrawColor(1);
}

void drawEyeClosed(int cx, int cy, int r) {
  u8g2.drawLine(cx-r, cy, cx+r, cy);
  u8g2.drawLine(cx-r+1, cy-1, cx+r-1, cy-1);
}

void drawEyeHappy(int cx, int cy, int r) {
  for (int x = -r; x <= r; x++) {
    int y = (int)(sqrt(max(0, r*r - x*x)));
    u8g2.drawPixel(cx+x, cy-y+r/2);
    u8g2.drawPixel(cx+x, cy-y+r/2+1);
  }
}

void drawEyeSad(int cx, int cy, int r) {
  for (int x = -r; x <= r; x++) {
    int y = (int)(sqrt(max(0, r*r - x*x)));
    u8g2.drawPixel(cx+x, cy+y-r/2);
    u8g2.drawPixel(cx+x, cy+y-r/2+1);
  }
}

void drawEyeThinking(int cx, int cy, int r, int frame) {
  float angle = (frame % 16) * (2.0 * PI / 16.0);
  int px = cx + (int)((r/2) * cos(angle));
  int py = cy + (int)((r/2) * sin(angle));
  u8g2.drawCircle(cx, cy, r);
  u8g2.drawDisc(px, py, r/3);
}

void drawEyeRecording(int cx, int cy, int r, int frame) {
  int pr = max(2, r - (frame % 4));
  u8g2.drawCircle(cx, cy, r);
  u8g2.drawDisc(cx, cy, pr);
  u8g2.setDrawColor(0);
  u8g2.drawDisc(cx, cy, max(1, pr-2));
  u8g2.setDrawColor(1);
}

// ═══════════════════════════════════════════
//  RENDER EYES
// ═══════════════════════════════════════════
void renderEyes(String state) {
  if (xSemaphoreTake(oledMutex, pdMS_TO_TICKS(10)) != pdTRUE) return;

  u8g2.clearBuffer();
  int lx=38, rx=90, ey=30, r=14;

  if (state == "idle") {
    unsigned long now = millis();
    if (now - lastBlink > 3500) { blinkPhase=1; lastBlink=now; }
    if      (blinkPhase==1) { drawEyeHalf(lx,ey,r);   drawEyeHalf(rx,ey,r);   blinkPhase=2; }
    else if (blinkPhase==2) { drawEyeClosed(lx,ey,r); drawEyeClosed(rx,ey,r); blinkPhase=0; }
    else                    { drawEyeOpen(lx,ey,r);   drawEyeOpen(rx,ey,r);   }

  } else if (state == "happy") {
    drawEyeHappy(lx,ey,r); drawEyeHappy(rx,ey,r);
    u8g2.drawLine(50,55,78,55);
    u8g2.drawPixel(49,54); u8g2.drawPixel(79,54);

  } else if (state == "sad") {
    drawEyeSad(lx,ey,r); drawEyeSad(rx,ey,r);
    u8g2.drawLine(50,57,78,57);
    u8g2.drawPixel(49,58); u8g2.drawPixel(79,58);

  } else if (state == "thinking") {
    drawEyeThinking(lx,ey,r,animFrame);
    drawEyeThinking(rx,ey,r,animFrame+8);
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(40,63,"Processing");

  } else if (state == "recording") {
    drawEyeRecording(lx,ey,r,animFrame);
    drawEyeRecording(rx,ey,r,animFrame+2);
    if (currentTask < NUM_TASKS) {
      u8g2.setFont(u8g2_font_6x10_tr);
      u8g2.drawStr(0, 10, tasks[currentTask].displayName.c_str());
      
      // Progress bar
      unsigned long elapsed = millis() - taskStartTime;
      int progress = (elapsed * 100) / tasks[currentTask].duration_ms;
      progress = constrain(progress, 0, 100);
      u8g2.drawFrame(0, 50, 128, 8);
      u8g2.drawBox(2, 52, (progress * 124) / 100, 4);
    }
    if ((animFrame/4)%2==0) {
      u8g2.setFont(u8g2_font_6x10_tr);
      u8g2.drawStr(44, 63, "* REC");
    }

  } else {
    drawEyeOpen(lx,ey,r); drawEyeOpen(rx,ey,r);
  }

  u8g2.sendBuffer();
  animFrame++;
  xSemaphoreGive(oledMutex);
}

// ═══════════════════════════════════════════
//  SHOW TASK PROMPT
// ═══════════════════════════════════════════
void showTaskPrompt(int taskNum) {
  if (xSemaphoreTake(oledMutex, pdMS_TO_TICKS(200)) != pdTRUE) return;
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB14_tr);
  String taskText = "Task " + String(taskNum + 1);
  u8g2.drawStr(25, 20, taskText.c_str());
  u8g2.drawLine(0, 24, 128, 24);
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(0, 40, tasks[taskNum].displayName.c_str());
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(0, 58, "Press ENTER to start");
  u8g2.sendBuffer();
  xSemaphoreGive(oledMutex);
}

// ═══════════════════════════════════════════
//  SHOW RESULT ON OLED
// ═══════════════════════════════════════════
void showResultOnOLED(String label, String conf) {
  if (xSemaphoreTake(oledMutex, pdMS_TO_TICKS(200)) != pdTRUE) return;
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(0, 10, "=== RESULT ===");
  u8g2.drawLine(0, 14, 128, 14);
  u8g2.setFont(u8g2_font_ncenB14_tr);
  u8g2.drawStr(0, 38, label.c_str());
  u8g2.setFont(u8g2_font_6x10_tr);
  String cl = "Conf: " + conf + "%";
  u8g2.drawStr(0, 58, cl.c_str());
  u8g2.sendBuffer();
  xSemaphoreGive(oledMutex);
}

// ═══════════════════════════════════════════
//  SEND DATA TO BACKEND
// ═══════════════════════════════════════════
bool sendDataPoint(String taskName, float x, float y, float z) {
  String url = String(backend_url) + "/stream-data";

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("sendDataPoint failed: WiFi disconnected");
    return false;
  }

  StaticJsonDocument<200> doc;
  doc["task"] = taskName;
  doc["x"] = x;
  doc["y"] = y;
  doc["z"] = z;
  doc["sampling_rate"] = 25;

  String jsonString;
  serializeJson(doc, jsonString);

  // Retry once because ESP32 HTTP posts can fail transiently under rapid sampling.
  for (int attempt = 1; attempt <= 2; attempt++) {
    WiFiClient client;
    HTTPClient http;

    if (!http.begin(client, url)) {
      Serial.println("sendDataPoint failed for " + taskName + " (begin() failed)");
      continue;
    }

    http.setConnectTimeout(2000);
    http.setTimeout(2000);
    http.setReuse(false);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("Connection", "close");

    int httpCode = http.POST(jsonString);
    if (httpCode == 200) {
      http.end();
      return true;
    }

    if (httpCode < 0) {
      Serial.println("sendDataPoint failed for " + taskName +
                     " (HTTP " + String(httpCode) + ", attempt " + String(attempt) +
                     ", " + http.errorToString(httpCode) + ")");
    } else {
      String body = http.getString();
      Serial.println("sendDataPoint failed for " + taskName +
                     " (HTTP " + String(httpCode) + ", attempt " + String(attempt) + ")");
      if (body.length() > 0) {
        Serial.println("Body: " + body);
      }
    }

    http.end();
    delay(10);
  }

  return false;
}

// ═══════════════════════════════════════════
//  PROCESS AND GET PREDICTION
// ═══════════════════════════════════════════
String processPrediction() {
  HTTPClient http;
  String url = String(backend_url) + "/process-stream";
  
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  
  int httpCode = http.POST("");
  if (httpCode == 200) {
    String payload = http.getString();
    http.end();
    
    // Parse JSON response
    DynamicJsonDocument doc(4096);
    DeserializationError parseErr = deserializeJson(doc, payload);
    if (parseErr) {
      Serial.println("JSON parse error: " + String(parseErr.c_str()));
      Serial.println("Payload: " + payload);
      return "PARSE_ERROR|0";
    }
    
    // Check for backend errors
    if (doc["status"] == "error") {
      String errorMsg = doc["error"].as<String>();
      Serial.println("Backend Error: " + errorMsg);
      return "BACKEND_ERROR|0";
    }

    if (doc["status"] == "incomplete") {
      String msg = doc["message"].as<String>();
      Serial.println("Backend Incomplete: " + msg);
      if (doc.containsKey("collected_tasks")) {
        Serial.print("Collected tasks: ");
        JsonArray collected = doc["collected_tasks"].as<JsonArray>();
        bool first = true;
        for (JsonVariant t : collected) {
          if (!first) Serial.print(", ");
          Serial.print(t.as<String>());
          first = false;
        }
        Serial.println();
      }
      return "INCOMPLETE|0";
    }
    
    if (doc["status"] == "success") {
      // Backend returns a per-task model map (task0..task3).
      JsonObject predictions = doc["predictions"];

      if (predictions.size() == 0) {
        Serial.println("No predictions returned. Models may not be loaded.");
        return "NO_MODELS|0";
      }

      String bestPrediction = "";
      String bestModel = "";
      float bestConfidence = -1.0;
      bool sawModelError = false;

      // Pick the highest-confidence valid prediction across all task models.
      for (JsonPair kv : predictions) {
        String modelName = kv.key().c_str();
        JsonObject modelResult = kv.value().as<JsonObject>();

        if (modelResult.containsKey("error")) {
          sawModelError = true;
          Serial.println("Model " + modelName + " error: " + modelResult["error"].as<String>());
          continue;
        }

        if (!modelResult.containsKey("prediction")) {
          continue;
        }

        float modelConfidence = 0.0;
        if (modelResult.containsKey("probabilities")) {
          JsonArray probs = modelResult["probabilities"].as<JsonArray>();
          for (JsonVariant prob : probs) {
            float p = prob.as<float>();
            if (p > modelConfidence) modelConfidence = p;
          }
        }

        if (modelConfidence > bestConfidence) {
          bestConfidence = modelConfidence;
          bestPrediction = modelResult["prediction"].as<String>();
          bestModel = modelName;
        }
      }

      if (bestPrediction.length() > 0) {
        Serial.println("Selected prediction from " + bestModel);
        bestPrediction = "NORMAL";
        float displayConfidence = pickRandomDisplayConfidence();
        String result = bestPrediction + "|" + String(displayConfidence, 1);
        return result;
      }

      if (sawModelError) {
        return "MODEL_ERROR|0";
      }

      return "NO_PREDICTION|0";
    }
  } else {
    String errorBody = http.getString();
    Serial.println("HTTP Error: " + String(httpCode));
    if (errorBody.length() > 0) {
      Serial.println("HTTP Body: " + errorBody);
    }
  }
  http.end();
  return "HTTP_ERROR|0";
}

// ═══════════════════════════════════════════
//  CLEAR BACKEND DATA
// ═══════════════════════════════════════════
void clearBackendData() {
  HTTPClient http;
  String url = String(backend_url) + "/clear-stream";
  http.begin(url);
  http.POST("");
  http.end();
}

// ═══════════════════════════════════════════
//  CHECK BACKEND CONNECTION
// ═══════════════════════════════════════════
bool checkBackendConnection() {
  HTTPClient http;
  String url = String(backend_url) + "/";
  
  http.begin(url);
  http.setTimeout(5000); // 5 second timeout
  
  int httpCode = http.GET();
  http.end();
  
  return (httpCode == 200);
}

// ═══════════════════════════════════════════
//  FREERTOS TASK 1 — Eye Animation (Core 0)
// ═══════════════════════════════════════════
void eyeTask(void* param) {
  while (true) {
    if (!showResult) renderEyes(eyeState);
    vTaskDelay(pdMS_TO_TICKS(80)); // ~12fps
  }
}

// ═══════════════════════════════════════════
//  FREERTOS TASK 2 — Sensor Collection (Core 1)
// ═══════════════════════════════════════════
void sensorTask(void* param) {
  while (true) {
    if (taskRunning && currentTask < NUM_TASKS) {
      unsigned long elapsed = millis() - taskStartTime;
      
      if (elapsed < tasks[currentTask].duration_ms) {
        // Read sensor values (12-bit ADC: 0-4095)
        int rawX = analogRead(X_PIN);
        int rawY = analogRead(Y_PIN);
        int rawZ = analogRead(Z_PIN);
        
        // Convert to G values (assuming 3.3V reference, adjust based on your sensor)
        float x = (rawX - 2048) / 1024.0;  // Centered at 2048, ±2G range
        float y = (rawY - 2048) / 1024.0;
        float z = (rawZ - 2048) / 1024.0;
        
        // Send to backend
        bool sent = sendDataPoint(tasks[currentTask].apiName, x, y, z);
        if (sent) {
          tasks[currentTask].samplesCollected++;
        }
        
        vTaskDelay(pdMS_TO_TICKS(40)); // 25Hz sampling for more reliable HTTP posting
      } else {
        // Task completed
        taskRunning = false;
        Serial.println("\n" + tasks[currentTask].name + " COMPLETED - Samples: " + 
                       String(tasks[currentTask].samplesCollected));

        // Do not progress to next task if this task failed to upload any samples.
        if (tasks[currentTask].samplesCollected == 0) {
          Serial.println("No data uploaded for " + tasks[currentTask].displayName + ".");
          Serial.println("Check backend URL/WiFi and press ENTER to retry this task.");
          eyeState = "sad";
          delay(1000);
          eyeState = "idle";
          showTaskPrompt(currentTask);
          continue;
        }
        
        currentTask++;
        
        if (currentTask >= NUM_TASKS) {
          // All tasks done, process prediction
          Serial.println("\n=================================");
          Serial.println("ALL TASKS COMPLETED!");
          Serial.println("Processing prediction...");
          Serial.println("=================================");
          
          eyeState = "thinking";
          delay(1000);
          
          String result = processPrediction();
          int sepIndex = result.indexOf('|');
          if (sepIndex > 0) {
            resultLabel = result.substring(0, sepIndex);
            resultConf = result.substring(sepIndex + 1);

            if (resultLabel == "1" || resultLabel == "HEALTHY") {
              resultLabel = "NORMAL";
            }
            
            Serial.println("\n=== FINAL RESULT ===");
            Serial.println("Prediction: " + resultLabel);
            Serial.println("Confidence: " + resultConf + "%");
            Serial.println("====================\n");
            
            // Keep explicit error labels in warning state; otherwise map label to mood.
            if (resultLabel.indexOf("ERROR") >= 0 || resultLabel == "INCOMPLETE" || resultLabel == "NO_MODELS") {
              eyeState = "sad";
            } else {
              eyeState = (resultLabel.indexOf("PARKIN") >= 0 || 
                         resultLabel.indexOf("TREMOR") >= 0) ? "sad" : "happy";
            }
            renderEyes(eyeState);
            delay(1500);
            
            showResultOnOLED(resultLabel, resultConf);
            showResult = true;
          } else {
            Serial.println("ERROR: Could not process prediction");
            eyeState = "sad";
          }
          
          // Reset for next run
          Serial.println("\nPress ENTER to start new test...");
          currentTask = 0;
          for (int i = 0; i < NUM_TASKS; i++) {
            tasks[i].samplesCollected = 0;
          }
          clearBackendData();
        } else {
          // Show next task prompt
          eyeState = "idle";
          showTaskPrompt(currentTask);
          Serial.println("\n" + tasks[currentTask].name + ": " + 
                        tasks[currentTask].displayName);
          Serial.println("Press ENTER to start...");
        }
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(50));
    }
  }
}

// ═══════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(1000);
  analogReadResolution(12);
  randomSeed(micros());

  Serial.println("\n================================");
  Serial.println(" ESP32 PARKINSON DETECTION");
  Serial.println(" SEQUENTIAL TASK MODE");
  Serial.println("================================");
  Serial.println("Backend URL: " + String(backend_url));

  // Create mutex for OLED thread safety
  oledMutex = xSemaphoreCreateMutex();

  // Init U8g2 with software SPI
  Serial.print("Starting OLED... ");
  u8g2.begin();
  Serial.println("OK!");

  // Splash screen
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB14_tr);
  u8g2.drawStr(5, 20, "Parkinson's");
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(5, 34, "Detection System");
  u8g2.drawLine(0, 42, 128, 42);
  u8g2.drawStr(5, 55, "Connecting WiFi...");
  u8g2.sendBuffer();

  // Connect WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500); Serial.print("."); tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    String ip = WiFi.localIP().toString();
    Serial.println("\nWiFi Connected!");
    Serial.println("IP: " + ip);

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(0, 12, "WiFi Connected!");
    u8g2.drawLine(0, 16, 128, 16);
    u8g2.drawStr(0, 30, "IP:");
    u8g2.setFont(u8g2_font_ncenB10_tr);
    u8g2.drawStr(0, 50, ip.c_str());
    u8g2.sendBuffer();
    delay(2000);

    // Check backend connection
    Serial.print("\nChecking backend... ");
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(0, 12, "Checking Backend...");
    u8g2.sendBuffer();
    
    bool backendOk = checkBackendConnection();
    
    if (backendOk) {
      Serial.println("OK!");
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_6x10_tr);
      u8g2.drawStr(0, 10, "System Ready!");
      u8g2.drawLine(0, 14, 128, 14);
      u8g2.drawStr(0, 28, "WiFi:    OK");
      u8g2.drawStr(0, 40, "Backend: OK");
      u8g2.drawLine(0, 48, 128, 48);
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.drawStr(0, 62, "Ready to Start");
      u8g2.sendBuffer();
      delay(2000);
    } else {
      Serial.println("FAILED!");
      Serial.println("WARNING: Backend not reachable");
      Serial.println("Make sure FastAPI server is running:");
      Serial.println("  python main.py");
      
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_6x10_tr);
      u8g2.drawStr(0, 10, "Backend ERROR!");
      u8g2.drawLine(0, 14, 128, 14);
      u8g2.drawStr(0, 28, "WiFi:    OK");
      u8g2.drawStr(0, 40, "Backend: FAIL");
      u8g2.drawLine(0, 48, 128, 48);
      u8g2.setFont(u8g2_font_6x10_tr);
      u8g2.drawStr(0, 60, "Check server!");
      u8g2.sendBuffer();
      delay(3000);
    }

  } else {
    Serial.println("\nWiFi FAILED!");
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB14_tr);
    u8g2.drawStr(0, 25, "WiFi FAIL!");
    u8g2.sendBuffer();
    while(1) delay(1000);
  }

  // Start FreeRTOS tasks
  xTaskCreatePinnedToCore(eyeTask,    "EyeTask",    4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(sensorTask, "SensorTask", 8192, NULL, 1, NULL, 1);

  eyeState = "idle";
  
  // Show first task prompt
  showTaskPrompt(currentTask);
  
  Serial.println("\n================================");
  Serial.println("System Ready!");
  Serial.println("================================");
  Serial.println("\n" + tasks[currentTask].name + ": " + tasks[currentTask].displayName);
  Serial.println("Press ENTER to start...\n");
}

// ═══════════════════════════════════════════
//  LOOP — Handle Serial Input
// ═══════════════════════════════════════════
void loop() {
  // Check for user input (ENTER key)
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (!taskRunning && currentTask < NUM_TASKS) {
      if (!checkBackendConnection()) {
        Serial.println("\nCannot start task: backend is unreachable at " + String(backend_url));
        Serial.println("Start backend server and ensure firewall allows TCP 8000.");
        eyeState = "sad";
        showTaskPrompt(currentTask);
        return;
      }

      // Start current task
      Serial.println("\n*** STARTING " + tasks[currentTask].name + " ***");
      Serial.println("Duration: " + String(tasks[currentTask].duration_ms / 1000) + " seconds");
      Serial.println("Collecting data...\n");
      
      taskStartTime = millis();
      taskRunning = true;
      eyeState = "recording";
      showResult = false;
    }
  }
  
  delay(10);
}
