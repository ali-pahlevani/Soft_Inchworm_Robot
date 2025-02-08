import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

# Load the Excel file
file_path = "../Data/Experimental Results/Experiment_Results_Finalized.xlsx"
df = pd.read_excel(file_path)

# Ensure output directory exists
output_dir = "../Images/Experimental_Test_Results"
os.makedirs(output_dir, exist_ok=True)

# Standardizing material names
df["Estimated Material"] = df["Estimated Material"].str.replace("Material_", "")

# 1️. Velocity Prediction Performance
mae_velocity = mean_absolute_error(df["Measured Velocity"], df["Calculated Velocity"])
mse_velocity = mean_squared_error(df["Measured Velocity"], df["Calculated Velocity"])
rmse_velocity = np.sqrt(mse_velocity)

# Velocity Error Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(df["Measured Velocity"] - df["Calculated Velocity"], bins=20, kde=True, color='blue')
plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
plt.title("Velocity Prediction Error Distribution")
plt.xlabel("Measured - Calculated Velocity")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "velocity_error_distribution.png"), dpi=300)
plt.show()

# 2️. Material Prediction Performance
material_report = classification_report(df["True Material"], df["Estimated Material"], output_dict=True)

# Convert classification report dictionary to DataFrame
material_report_df = pd.DataFrame(material_report).transpose()

# Round the values to 2 decimal places
material_report_df = material_report_df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

# Generate the Material Classification Report as a table using Matplotlib
def create_table_from_df(df, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5])

    # Make header bold
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(fontweight="bold")

    # Save the table as an image
    #table_path = os.path.join(output_dir, filename)
    #plt.savefig(table_path, dpi=300, bbox_inches="tight")
    #plt.show()
    #print(f"Table saved to {table_path}")

# Save the Material Classification Report
#create_table_from_df(material_report_df, "material_classification_report.png")

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(df["True Material"], df["Estimated Material"])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df["True Material"].unique(), yticklabels=df["True Material"].unique())
plt.xlabel("Predicted Material")
plt.ylabel("True Material")
plt.title("Confusion Matrix of Material Classification")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.show()

# 3️. Material-wise Performance Bar Plot
df["Material Correct"] = (df["Estimated Material"] == df["True Material"]).astype(int)
accuracy_per_material = df.groupby("True Material")["Material Correct"].mean() * 100

plt.figure(figsize=(8, 5))
bar_plot = sns.barplot(x=accuracy_per_material.index, y=accuracy_per_material.values, palette='viridis')
plt.title("Material-wise Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)

# Add percentage annotations above each bar
for p in bar_plot.patches:
    bar_plot.annotate(
        f'{p.get_height():.2f}%', 
        (p.get_x() + p.get_width() / 2., p.get_height() / 1.02), 
        ha='center', va='center', 
        fontsize=10, color='black', fontweight='bold', 
        xytext=(0, 10), textcoords='offset points'
    )

# Save the plot with annotations
plt.savefig(os.path.join(output_dir, "material_accuracy.png"), dpi=300)
plt.show()

# 4️. Mean Optimal Pressure and Frequency
mean_optimal_pressure = df["Optimal Pressure"].mean()
mean_optimal_frequency = df["Optimal Frequency"].mean()

# Create a DataFrame for summary metrics (rounded to 3 decimals)
summary_data = {
    "Metric": ["Mean Optimal Pressure", "Mean Optimal Frequency"],
    "Value": [round(mean_optimal_pressure, 3), round(mean_optimal_frequency, 3)],
}
summary_df = pd.DataFrame(summary_data)

# Generate the summary table as an image using Matplotlib
create_table_from_df(summary_df, "summary_table.png")

# 5️. Residual Velocity Analysis Box Plot
df["Residual Velocity"] = df["Measured Velocity"] - df["Calculated Velocity"]
plt.figure(figsize=(8, 5))
sns.boxplot(x="True Material", y="Residual Velocity", data=df, palette='coolwarm')
plt.title("Residual Velocity Analysis by Material")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "residual_velocity.png"), dpi=300)
plt.show()

# 6️. Save results to CSV
df.to_csv("../Data/Experimental Results/analyzed_results.csv", index=False)
print("\nResults saved to 'analyzed_results.csv'")
