import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Load the data from an Excel file
data_path = "../Data/dataReady.xlsx"
df = pd.read_excel(data_path)

# Check for missing values in the dataset
if df.isnull().any().any():
    print("Dataset contains missing values. Filling missing values...")
    df.fillna(df.mean(), inplace=True)

# Map materials to their friction coefficients
material_to_coeff = {
    "Rubber": 0.58,
    "Paper": 1.19,
    "Plastic": 0.84,
    "Iron": 0.27,
    "Glass": 2.75,
}

df["Friction_Coeff"] = df["Material"].map(material_to_coeff)

# Remove rows with NaN after mapping
if df["Friction_Coeff"].isnull().any():
    print("Removing rows with unmapped materials...")
    df = df.dropna(subset=["Friction_Coeff"])

# Data augmentation
if len(df) < 1000:
    num_augmentations = 5  # Number of times you want to augment the data
    augmented_data = []
    for _ in range(num_augmentations):  # Loop to augment the data multiple times
        for _, row in df.iterrows():
            numeric_part = row[["Pressure", "Frequency", "Velocity"]].astype(float).values
            noise = np.random.normal(0, 0.01, size=len(numeric_part))
            augmented_numeric = numeric_part + noise
            if np.any(np.isnan(augmented_numeric)) or np.any(np.isinf(augmented_numeric)):
                continue  # Skip invalid rows
            augmented_row = row.copy()
            augmented_row[["Pressure", "Frequency", "Velocity"]] = augmented_numeric
            augmented_data.append(augmented_row)

    # Concatenate augmented data to the original dataframe
    df = pd.concat([df] + [pd.DataFrame(augmented_data)] * num_augmentations, ignore_index=True)

# Shuffle the dataset
df = shuffle(df).reset_index(drop=True)

# Encode the "Material" column with one-hot encoding
material_dummies = pd.get_dummies(df["Material"], prefix="Material")
X = pd.concat([df[["Pressure", "Frequency"]], material_dummies], axis=1).values
y = df["Velocity"].values

# Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the Neural Network model
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

# Initialize the model with the correct input size (based on one-hot encoded material features)
input_size = X_train.shape[1]
model = NeuralNetwork(input_size=input_size)

# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop with mini-batching
batch_size = 64
num_epochs = 1500
best_val_loss = float('inf')
num_batches = len(X_train_tensor) // batch_size
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()

    # Iterate through batches
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        X_batch = X_train_tensor[batch_start:batch_end]
        y_batch = y_train_tensor[batch_start:batch_end]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Check for NaN loss and debug
        if torch.isnan(loss):
            print("NaN detected in loss. Debugging...")
            print("Sample Inputs:", X_batch[:5])
            print("Sample Outputs:", outputs[:5])
            break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "Trained_Model/best_model.pth")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

print(f"Best Validation Loss: {best_val_loss:.4f}")

# Save the model architecture + weights, scaler, and material mappings for later use
torch.save(model.state_dict(), "Trained_Model/best_model_weights.pth")
joblib.dump(scaler, 'Trained_Model/scaler.pkl')
joblib.dump(material_dummies, 'Trained_Model/material_dummies.pkl')
joblib.dump(input_size, 'Trained_Model/input_size.pkl')

# Load the best model for evaluation
import os
if os.path.exists("Trained_Model/best_model.pth"):
    model.load_state_dict(torch.load("Trained_Model/best_model.pth"))
else:
    print("Model file not found. Ensure the training process is successful.")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy()
    test_loss = mean_squared_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    print(f"Test MSE: {test_loss:.4f}, R2 Score: {r2:.4f}")

######### Visualization #########
# Visualization - True vs Predicted Values
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, label="True Values", alpha=0.7)
plt.scatter(range(len(test_predictions)), test_predictions, label="Predictions", alpha=0.7)
plt.legend()
plt.title("True vs Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Velocity")
plt.grid()
plt.savefig('../Images/Model_Train_Images/Train_Result.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization - Residual Histogram
sns.histplot(y_test - test_predictions.flatten(), kde=True, bins=30)
plt.title("Residual Histogram")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid()
plt.savefig('../Images/Model_Train_Images/Train_Histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization - Loss Curve (Training & Validation Loss)
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", alpha=0.7)
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", alpha=0.7)
plt.legend()
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig('../Images/Model_Train_Images/Loss_Curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization - True vs Predicted Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(y_test, label="True Values", color="blue", shade=True, alpha=0.5)
sns.kdeplot(test_predictions.flatten(), label="Predicted Values", color="red", shade=True, alpha=0.5)
plt.legend()
plt.title("True vs Predicted Value Distribution")
plt.xlabel("Velocity")
plt.ylabel("Density")
plt.grid()
plt.savefig('../Images/Model_Train_Images/True_vs_Predicted_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()
#################################

############ Metrics ############
threshold = 0.5
y_test_binary = (y_test > threshold).astype(int)
test_predictions_binary = (test_predictions.flatten() > threshold).astype(int)

# Calculate metrics
test_loss = mean_squared_error(y_test, test_predictions)  # MSE
test_rmse = np.sqrt(test_loss)  # RMSE
test_mae = mean_absolute_error(y_test, test_predictions)  # MAE
r2 = r2_score(y_test, test_predictions)  # R²
adjusted_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))  # Adjusted R²

precision = precision_score(y_test_binary, test_predictions_binary, zero_division=1)
recall = recall_score(y_test_binary, test_predictions_binary, zero_division=1)
f1 = f1_score(y_test_binary, test_predictions_binary, zero_division=1)
accuracy = accuracy_score(y_test_binary, test_predictions_binary)

# Create a DataFrame for metrics (rounded to 3 decimal places)
metrics_data = {
    "Metric": [
        "MSE", "RMSE", "MAE", "R²", "Adjusted R²", 
        "Precision", "Accuracy", "Recall", "F1-Score"
    ],
    "Value": [
        round(test_loss, 3), round(test_rmse, 3), round(test_mae, 3), 
        round(r2, 3), round(adjusted_r2, 3), 
        round(precision, 3), round(accuracy, 3), 
        round(recall, 3), round(f1, 3)
    ],
}
metrics_df = pd.DataFrame(metrics_data)

# Generate the plain table as an image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("tight")
ax.axis("off")

# Create the table directly from the DataFrame
table = ax.table(
    cellText=metrics_df.values,
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="center",
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width([0, 1])

# Make header bold
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_text_props(fontweight="bold")

# Save the table as an image
metrics_table_path = "../Images/Model_Train_Images/metrics_table_plain.png"
plt.savefig(metrics_table_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Metrics table saved to {metrics_table_path}")
#################################

# Save the preprocessed data as CSV
preprocessed_data_path = "../Data/preprocessed_data.csv"
df.to_csv(preprocessed_data_path, index=False)
print(f"Preprocessed data saved to {preprocessed_data_path}")

# Example Inference
def predict_velocity(pressure, frequency, material_name):
    # Map material name to one-hot encoding
    material_vector = np.zeros(len(material_dummies.columns))
    if f"Material_{material_name}" in material_dummies.columns:
        material_vector[material_dummies.columns.get_loc(f"Material_{material_name}")] = 1
    else:
        raise ValueError(f"Material '{material_name}' is not recognized. Add it to the material map.")

    # Combine inputs
    input_vector = np.hstack([pressure, frequency, material_vector])
    input_normalized = scaler.transform([input_vector])
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32)

    # Predict using the model
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# Test inference
example_pressure = 150
example_frequency = 0.125
example_material = "Rubber"
predicted_velocity = predict_velocity(example_pressure, example_frequency, example_material)
print(f"Predicted Velocity for {example_material}: {predicted_velocity:.4f}")
