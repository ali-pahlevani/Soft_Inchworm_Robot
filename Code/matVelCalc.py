import torch
import joblib
import numpy as np
import time
import csv
from torch import nn

# Define the Neural Network class (this should match the one in the training script)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# --- Load Model and Associated Files ---
def load_model_and_files():
    input_size = joblib.load(open('Trained_Model/input_size.pkl', 'rb'))
    model = NeuralNetwork(input_size=input_size)
    weights = torch.load('Trained_Model/best_model_weights.pth')
    model.load_state_dict(weights)
    scaler = joblib.load(open('Trained_Model/scaler.pkl', 'rb'))
    material_dummies = joblib.load(open('Trained_Model/material_dummies.pkl', 'rb'))
    materials = list(material_dummies.keys())
    return model, scaler, materials


# --- Estimate Material Based on Observed Velocity ---
def estimate_material(model, scaler, materials, observed_velocity, current_pressure, current_frequency):
    min_diff = float("inf")
    estimated_material = None
    for material in materials:
        material_vector = np.zeros(len(materials))
        material_index = materials.index(material)
        material_vector[material_index] = 1
        input_vector = np.hstack([current_pressure, current_frequency, material_vector])
        input_scaled = scaler.transform([input_vector])
        with torch.no_grad():
            predicted_velocity = model(torch.tensor(input_scaled, dtype=torch.float32)).item()
        diff = abs(predicted_velocity - observed_velocity)
        if diff < min_diff:
            min_diff = diff
            estimated_material = material
    return estimated_material


# --- Optimize Inputs for Maximum Velocity ---
def optimize_inputs(model, scaler, materials, estimated_material, pressure_range, frequency_range):
    max_velocity = float("-inf")
    optimal_pressure, optimal_frequency = None, None
    previous_max_velocity = float("-inf")
    for pressure in sorted(pressure_range):  # Start from the lowest pressure
        for frequency in sorted(frequency_range):  # Start from the lowest frequency
            material_vector = np.zeros(len(materials))
            material_index = materials.index(estimated_material)
            material_vector[material_index] = 1
            input_vector = np.hstack([pressure, frequency, material_vector])
            input_scaled = scaler.transform([input_vector])
            with torch.no_grad():
                predicted_velocity = model(torch.tensor(input_scaled, dtype=torch.float32)).item()
            if predicted_velocity > max_velocity and predicted_velocity >= 1.1 * previous_max_velocity:
                max_velocity = predicted_velocity
                optimal_pressure, optimal_frequency = pressure, frequency
                previous_max_velocity = max_velocity
    return optimal_pressure, optimal_frequency


# --- Main Real-Time Loop ---
def main():
    model, scaler, materials = load_model_and_files()
    pressure_range = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    frequency_range = [0.125, 0.17, 0.25, 0.33, 0.4]
    loop_counter = 0

    with open("../Data/Results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Input Pressure", "Input Frequency", "Measured Velocity", "Calculated Velocity", "Estimated Material", "Optimal Pressure", "Optimal Frequency", "Estimated Max Velocity"])

    while True:
        try:
            loop_counter += 1
            print(f"\nTest {loop_counter}")
            print("-" * 40)
            
            while True:
                try:
                    current_pressure = int(input(f"Enter input pressure {pressure_range}: "))
                    if current_pressure in pressure_range:
                        break
                    else:
                        print("Invalid selection. Please choose a valid pressure.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            while True:
                try:
                    current_frequency = float(input(f"Enter input frequency {frequency_range}: "))
                    if current_frequency in frequency_range:
                        break
                    else:
                        print("Invalid selection. Please choose a valid frequency.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            measured_velocity = float(input("Enter measured velocity: "))
            estimated_material = estimate_material(model, scaler, materials, measured_velocity, current_pressure, current_frequency)
            optimal_pressure, optimal_frequency = optimize_inputs(model, scaler, materials, estimated_material, pressure_range, frequency_range)
            
            with torch.no_grad():
                material_vector = np.zeros(len(materials))
                material_index = materials.index(estimated_material)
                material_vector[material_index] = 1
                input_vector = np.hstack([current_pressure, current_frequency, material_vector])
                input_scaled = scaler.transform([input_vector])
                calculated_velocity = model(torch.tensor(input_scaled, dtype=torch.float32)).item()
                
                optimal_input_vector = np.hstack([optimal_pressure, optimal_frequency, material_vector])
                optimal_input_scaled = scaler.transform([optimal_input_vector])
                estimated_max_velocity = model(torch.tensor(optimal_input_scaled, dtype=torch.float32)).item()
            
            print(f"Estimated Material: {estimated_material}")
            print(f"Optimal Pressure: {optimal_pressure}, Optimal Frequency: {optimal_frequency}")
            print(f"Estimated Maximum Velocity: {estimated_max_velocity}")
            
            with open("../Data/Results.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([current_pressure, current_frequency, measured_velocity, calculated_velocity, estimated_material, optimal_pressure, optimal_frequency, estimated_max_velocity])
            
            time.sleep(3)
        
        except ValueError:
            print("Invalid data received or entered. Please try again.")
        except KeyboardInterrupt:
            print("Exiting program.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
