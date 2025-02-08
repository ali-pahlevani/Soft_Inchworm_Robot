const int sensorPin = 2;  // Sensor output connected to A0
const float stripeDistance = 0.01; // Distance between stripes in meters (m)
const int numReadings = 5; // Number of readings to calculate the mean velocity

int sensorValue = 0;
int threshold = 500; // Adjust based on calibration
unsigned long lastTime = 0;
unsigned long currentTime = 0;

float totalDistance = 0.0;   // Accumulated distance traveled
unsigned long totalTime = 0; // Accumulated time in milliseconds
int measurementsCount = 0;   // Counter for measurements

void setup() {
  Serial.begin(115200);
  pinMode(sensorPin, INPUT);
}

void loop() {
  sensorValue = digitalRead(sensorPin);

  // If a black line is detected (based on threshold)
  if (sensorValue < threshold) {
    currentTime = millis(); // Record the time when the black line is detected

    if (lastTime != 0) { // Avoid calculating velocity on the first measurement
      // Calculate time difference between two detections in milliseconds
      unsigned long timeDiff = currentTime - lastTime;
      
      // Accumulate total distance and total time
      totalDistance += stripeDistance;
      totalTime += timeDiff;

      measurementsCount++;

      // After 'n' measurements, calculate mean velocity
      if (measurementsCount >= numReadings) {
        // Calculate mean velocity in meters per second
        float meanVelocity = totalDistance / (totalTime / 1000.0); // Convert time to seconds

        // Print the result
        //Serial.print("Mean Velocity: ");
        Serial.print(meanVelocity);

        //Serial.print("Measurement: ");
        //Serial.print(sensorValue);
        //Serial.println(" ");

        //Serial.println(" m/s");
        // Reset for next calculation
        totalDistance = 0.0;
        totalTime = 0;
        measurementsCount = 0;
      }
    }

    lastTime = currentTime; // Update the lastTime for the next transition
    //delay(100); // Add a small delay to debounce and avoid multiple readings for the same stripe
  }
  Serial.print("Measurement: ");
  Serial.print(sensorValue);
  Serial.println(" ");
  delay(200);
}
