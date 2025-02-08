import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error

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

# Load the saved model, scaler, and material mappings
model = NeuralNetwork(input_size=joblib.load('Trained_Model/input_size.pkl'))
model.load_state_dict(torch.load('Trained_Model/best_model_weights.pth'))
scaler = joblib.load('Trained_Model/scaler.pkl')
material_dummies = joblib.load('Trained_Model/material_dummies.pkl')

# Function to preprocess the new data (similar to training)
def preprocess_new_data(data_path):
    df = pd.read_excel(data_path)

    # Map materials to their friction coefficients (same as during training)
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

    # Encode the "Material" column with one-hot encoding (same as training)
    material_dummies_new = pd.get_dummies(df["Material"], prefix="Material")
    X_new = pd.concat([df[["Pressure", "Frequency"]], material_dummies_new], axis=1).values

    # Normalize the features using the saved scaler
    X_new_normalized = scaler.transform(X_new)

    return X_new_normalized, df

# Load and preprocess the new data
data_path = "../Data/dataReady.xlsx"
X_new_normalized, df = preprocess_new_data(data_path)

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new_normalized, dtype=torch.float32)

# Predict using the loaded model
model.eval()
with torch.no_grad():
    predictions = model(X_new_tensor).numpy()

# Add predictions to the dataframe
df["Predicted_Velocity"] = predictions.flatten()

######### Visualization #########
# Visualization 1: True vs Predicted Values
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df["Velocity"], label="True Values", alpha=0.7)
plt.scatter(df.index, df["Predicted_Velocity"], label="Predictions", alpha=0.7)
plt.legend()
plt.title("True vs Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Velocity (cm/s)")
plt.grid()
plt.savefig('../Images/Model_Test_Images/Test_Result.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Residual Histogram
sns.histplot(df["Velocity"] - df["Predicted_Velocity"], kde=True, bins=30)
plt.title("Residual Histogram")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid()
plt.savefig('../Images/Model_Test_Images/Test_Histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: True vs Predicted Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(df["Velocity"], label="True Values", color="blue", shade=True, alpha=0.5)
sns.kdeplot(df["Predicted_Velocity"], label="Predicted Values", color="red", shade=True, alpha=0.5)
plt.legend()
plt.title("True vs Predicted Value Distribution")
plt.xlabel("Velocity (cm/s)")
plt.ylabel("Density")
plt.grid()
plt.savefig('../Images/Model_Test_Images/True_vs_Predicted_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()
#################################

############ Metrics ############
# Binary classification threshold
threshold = 0.5
y_true = df["Velocity"].values
y_pred = df["Predicted_Velocity"].values
y_true_binary = (y_true > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
adjusted_r2 = 1 - (1 - r2) * ((len(y_true) - 1) / (len(y_true) - X_new_tensor.shape[1] - 1))

precision = precision_score(y_true_binary, y_pred_binary, zero_division=1)
recall = recall_score(y_true_binary, y_pred_binary, zero_division=1)
f1 = f1_score(y_true_binary, y_pred_binary, zero_division=1)
accuracy = accuracy_score(y_true_binary, y_pred_binary)

# Create a DataFrame for metrics (rounded to 3 decimals)
metrics_data = {
    "Metric": [
        "MSE", "RMSE", "MAE", "R²", "Adjusted R²", 
        "Precision", "Accuracy", "Recall", "F1-Score"
    ],
    "Value": [
        round(mse, 3), round(rmse, 3), round(mae, 3), 
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
metrics_table_path = "../Images/Model_Test_Images/metrics_table_plain.png"
plt.savefig(metrics_table_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Metrics table saved to {metrics_table_path}")
#################################

# Save the results with predictions to a new Excel file
output_path = "../Data/predictions_with_results.xlsx"
df.to_excel(output_path, index=False)
print(f"Predictions and results saved to {output_path}")
