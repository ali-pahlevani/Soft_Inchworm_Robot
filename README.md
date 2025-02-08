# Soft_Inchworm_Robot


<div><img src="https://github.com/user-attachments/assets/01d786ea-7403-40ad-9667-27592337a50d" alt="Soft_Inchworm_Robot" width="800" height="500"/></div>

### Modeling:
Modeling the behavior of soft inchworm robots using data-driven approaches is essential for understanding their motion dynamics and optimizing their performance. To predict the robot's velocity based on key operational parameters, a neural network model is employed, utilizing data features such as pressure, frequency, and material type. The approach incorporates several stages, including data preprocessing, neural network design, training, and evaluation, with the ultimate goal of developing a predictive model capable of estimating the velocity of the robot.

### Data Preprocessing
The raw dataset, sourced from an Excel file, undergoes extensive preprocessing to ensure it is suitable for use in the neural network model. Initially, any missing values in the dataset are handled by filling them with the mean of the respective columns. This is followed by the mapping of the categorical "Material" feature to corresponding friction coefficients, transforming it into a numerical value. This feature, labeled "Friction_Coeff," is integrated into the dataset, providing a continuous variable for the neural network to learn from.
Data augmentation is employed to enhance the dataset when enough samples are not available. Small Gaussian noise is added to the numeric features—pressure, frequency, and velocity—thus increasing the dataset's size and variability. This augmented data is combined with the original dataset, and the resulting data is shuffled to avoid any sequence bias.
Additionally, the "Material" feature is encoded using one-hot encoding, creating binary columns for each material type. This transformation enables the neural network to process categorical data numerically. The feature set is then normalized using Min-Max scaling, ensuring all input features fall within the same range, which is critical for the network to treat them with equal importance during training.

### Neural Network Model Architecture
A feed-forward neural network is employed to model the relationship between the input features (pressure, frequency, and material type) and the target variable (velocity). The network consists of multiple fully connected layers, each followed by a ReLU activation function to introduce non-linearity and enhance the network's ability to capture complex relationships. To mitigate overfitting, dropout regularization is applied to the hidden layers, randomly deactivating a fraction of the neurons during training.
The input to the network includes the preprocessed features, which are fed through the model to predict a single output: the velocity. The output layer consists of a single neuron, corresponding to the predicted velocity. The model is trained using the Mean Squared Error (MSE) loss function, which quantifies the difference between the predicted and true velocity values. The Adam optimizer is used to minimize the MSE loss function and adjust the model’s parameters during training.

### Model Training and Evaluation
The dataset is divided into training, validation, and test sets to enable rigorous model training and evaluation. The training set is used to optimize the model’s parameters, while the validation set monitors the model's performance during training and selects the best model based on the lowest validation loss. The test set, which is kept separate from the training and validation sets, serves as the final evaluation benchmark.
Training occurs over 1500 epochs, with the model processing mini-batches of 64 samples per epoch. **Fig. 1** illustrates the loss curve during training, showing the progression of training and validation loss over epochs:

<div><img src="https://github.com/user-attachments/assets/ccad6cdb-fe23-4ac4-90c1-60e57a44f77d" alt="Fig. 1" width="700" height="400"/></div>

At each epoch, the model's performance on the validation set is evaluated, and the weights are updated accordingly. To maintain stable training, gradient clipping is applied to prevent the occurrence of exploding gradients. The model with the best validation performance (i.e., the lowest validation loss) is saved for later use.
After training, the model is evaluated on the test set using several performance metrics, such as Mean Squared Error (MSE), R² score, and others, to assess its generalization ability and predictive accuracy. **Table 1** summarizes the key evaluation metrics, including MSE, RMSE, and precision, providing a quantitative assessment of the model’s performance:

<div><img src="https://github.com/user-attachments/assets/ecbff0f8-e3ca-43f2-a3e2-9447c2fceeb9" alt="Table 1" width="550" height="300"/></div>

To analyze the model’s prediction errors, **Fig. 2** presents a residual histogram, highlighting the distribution of residuals (differences between true and predicted values):

<div><img src="https://github.com/user-attachments/assets/6de45d31-ebea-43f7-be02-db031a645eda" alt="Fig. 2" width="600" height="400"/></div>

Additionally, **Fig. 3** compares the true velocity values versus the model’s predicted values on the test set, while **Fig. 4** shows the distributions of true and predicted values, offering insight into the overall prediction behavior:

<div><img src="https://github.com/user-attachments/assets/60c9118d-aea8-487c-86ea-5a25a7f9a396" alt="Fig. 3" width="700" height="400"/></div>

<div><img src="https://github.com/user-attachments/assets/e744fd61-006b-4461-950c-8f5ad6d1f755" alt="Fig. 4" width="700" height="400"/></div>

---

# Surface Material Identification and Adaptive Velocity Optimization:
After training the neural network model to predict the velocity of the inchworm robot, the next phase involves deploying the model for real-time surface material estimation and adaptive velocity optimization. This stage is crucial for enabling adaptive control, where the robot can dynamically adjust its operating conditions based on environmental factors. The algorithm is designed to estimate the material properties of the surface based on observed velocity and subsequently determine the optimal pressure and frequency settings to maximize the robot’s speed.

### Model Loading and Setup
To maintain consistency with the training phase, the deployed system first reconstructs the neural network architecture using the same structure defined during training. The model is initialized with saved weights from the training phase to ensure it retains its predictive capabilities. Additionally, precomputed data transformations such as input scaling (using a previously stored scaler) and one-hot encoded material representations are loaded. These transformations are essential for processing real-time inputs in a manner identical to the training phase, allowing the model to generate accurate velocity predictions.
The model relies on three key input components: the applied pressure, actuation frequency, and surface material. Since the material type is categorical, it is represented as a one-hot encoded vector. The system retrieves the set of known material types from a stored dictionary and constructs feature vectors accordingly. These feature vectors are then normalized using the same scaler used during training to ensure consistency.

### Material Estimation Based on Observed Velocity
One of the core functionalities of the deployed system is estimating the material of the surface on which the robot is operating. This estimation is performed by analyzing the relationship between input conditions (pressure and frequency) and the measured velocity of the robot. Given a user-provided input pressure, actuation frequency, and observed velocity, the algorithm systematically evaluates all possible material types.
For each candidate material, the system constructs a corresponding feature vector that includes the one-hot encoded material representation along with the applied pressure and frequency. This vector is transformed using the stored scaler and passed through the trained neural network to obtain a predicted velocity. The estimated material is determined by selecting the material whose predicted velocity is closest to the observed velocity. This method leverages the model’s learned relationships between materials and velocity patterns, enabling accurate identification of the surface properties in real time.

### Input Optimization for Maximum Velocity
Once the material type is estimated, the system proceeds to optimize the input conditions to maximize the robot’s velocity. This optimization is crucial for enhancing locomotion efficiency, especially in applications requiring adaptive control based on varying terrain properties.
The optimization process evaluates different combinations of pressure and frequency within predefined operational ranges. For each combination, a feature vector is constructed, scaled, and fed into the neural network model to predict the expected velocity. The optimal input conditions are identified by selecting the pressure and frequency pair that yields the highest predicted velocity while ensuring a stable performance gain.
To prevent excessive parameter fluctuations and maintain control stability, the algorithm enforces a constraint where new velocity predictions must exceed at least 1.1 times the previously recorded maximum before updating the optimal input selection. This constraint ensures that the optimization process results in meaningful improvements rather than minor variations due to model uncertainty.

### Interactive User Input and Real-Time Processing
The system operates within an interactive loop, allowing users to input pressure and frequency values while continuously updating estimations and optimizations. The interactive nature of this process enables real-time adaptation, allowing users to adjust parameters based on live performance feedback.
Each iteration follows a structured sequence:

**1.	User Input Collection:** The user is prompted to enter the current pressure and actuation frequency. The observed velocity is measured using a QRE1113 Reflectance Sensor connected to an Arduino Uno. The sensor detects black stripes on a moving surface, and velocity is calculated based on the time intervals between stripe detections. The measured velocity is then read via a serial interface and fed into the main Python code.

**2.	Material Estimation:** The system predicts the most likely material by comparing the measured velocity with model-predicted velocities for all known materials.

**3.	Optimization Execution:** The model determines the optimal pressure and frequency values that maximize the estimated velocity for the identified material.

**4.	Result Storage and Display:** The estimated material, optimal input conditions, and predicted maximum velocity are displayed to the user and stored in a CSV file for future analysis.
   
The real-time nature of this system enables continuous refinement of the robot’s control parameters based on changing surface conditions, improving locomotion efficiency in diverse environments.

---

# Experimental Results:
The proposed framework, comprising material estimation, velocity prediction, and input optimization, was tested on a variety of surfaces to assess its accuracy and effectiveness. The evaluation focuses on three key aspects: material classification performance, velocity prediction accuracy, and the overall system’s adaptability to different surface conditions.

### Surface Material Classification Performance
A crucial aspect of the system is its ability to accurately estimate the material type based on observed velocity. The confusion matrix in **Fig. 5** presents the classification results across different surface materials, highlighting both correctly identified cases and misclassifications:

<div><img src="https://github.com/user-attachments/assets/782f3cff-25f9-4325-8dc3-29cad86e3f6d" alt="Fig. 5" width="700" height="400"/></div>

The model achieves high accuracy in most cases, demonstrating its capability to distinguish material-dependent velocity patterns.
To provide a more detailed breakdown, **Fig. 6** illustrates the classification accuracy for each material type, showing how well the model distinguishes between different surfaces:

<div><img src="" alt="Fig. 6" width="700" height="400"/></div>

The results indicate that while classification performance remains high, some materials exhibit slightly lower accuracy due to overlapping velocity distributions.

### Velocity Prediction Accuracy
In addition to material classification, the framework’s ability to predict the robot’s velocity under different conditions was analyzed. **Fig. 7** presents a residual velocity analysis, displaying the distribution of prediction errors across various surface materials:

<div><img src="" alt="Fig. 7" width="700" height="400"/></div>

The results indicate that while the model maintains a low error margin for most surfaces, certain materials introduce larger variations, suggesting the potential influence of unmodeled factors.
Furthermore, **Fig. 8** shows the overall velocity prediction error distribution, providing insight into the deviations between true and predicted velocities:

<div><img src="" alt="Fig. 8" width="700" height="400"/></div>

The error remains minimal for most test cases, reinforcing the model’s reliability. However, minor discrepancies highlight areas where further refinements—such as integrating additional environmental features—could enhance performance.

### Overall Framework Assessment
Beyond individual model performance, the complete framework—including material estimation, input optimization, and real-time processing—was evaluated in a dynamic testing environment. The system successfully demonstrated:

**•	Real-time material estimation**, where the algorithm effectively identified the surface type based on velocity feedback.

**•	Adaptive input optimization**, dynamically adjusting pressure and frequency to maximize locomotion speed.

**•	Stable and efficient control**, ensuring that parameter adjustments led to meaningful performance improvements without excessive fluctuations.

The integration of machine learning and adaptive control proved effective in enhancing the robot’s motion across different terrains.

