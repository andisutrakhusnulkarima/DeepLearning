import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (Replace with actual dataset file path)
df = pd.read_csv("dataset.csv")  # Assuming dataset is in CSV format

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

# Ensure categorical columns exist before encoding
categorical_columns = ['Gender', 'Age', 'Ethnicity']
categorical_columns = [col for col in categorical_columns if col in df.columns]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Select features and target
X = df.drop(columns=['aveOralM'])
y = df['aveOralM']

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)).flatten()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors (PyTorch)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define Improved PyTorch Model with Hyperparameter Tuning
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Increased units
        self.fc2 = nn.Linear(512, 256)        # Increased units
        self.fc3 = nn.Linear(256, 128)        # Increased units
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)        # Adjusted dropout rate to 0.3
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

# Initialize model with increased complexity
input_dim = X_train.shape[1]
model = NeuralNetwork(input_dim)

# Loss and optimizer with tuned learning rate
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adjusted learning rate and weight decay

# Train model with tuned hyperparameters
num_epochs = 500  # Increased epochs for better training
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:  # Print every 50 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predictions (ensure that model is in evaluation mode during inference)
model.eval()
y_pred_pytorch = model(X_test_tensor).detach().numpy()

# TensorFlow Model with Hyperparameter Tuning
tf_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # Increased units
    keras.layers.Dropout(0.3),  # Adjusted dropout rate to 0.3
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

tf_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')  # Adjusted learning rate

# Train TensorFlow model with early stopping
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience
tf_model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=0, validation_data=(X_test, y_test), callbacks=[es])

# Predictions
y_pred_tf = tf_model.predict(X_test)

# Inverse transform results
y_pred_pytorch = scaler_y.inverse_transform(y_pred_pytorch)
y_pred_tf = scaler_y.inverse_transform(y_pred_tf)
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
mse_pytorch = mean_squared_error(y_test, y_pred_pytorch)
rmse_pytorch = np.sqrt(mse_pytorch)
r2_pytorch = r2_score(y_test, y_pred_pytorch)

mse_tf = mean_squared_error(y_test, y_pred_tf)
rmse_tf = np.sqrt(mse_tf)
r2_tf = r2_score(y_test, y_pred_tf)

# Print evaluation results
print("PyTorch Model Evaluation:")
print(f'MSE: {mse_pytorch:.4f}, RMSE: {rmse_pytorch:.4f}, R2 Score: {r2_pytorch:.4f}')

print("TensorFlow Model Evaluation:")
print(f'MSE: {mse_tf:.4f}, RMSE: {rmse_tf:.4f}, R2 Score: {r2_tf:.4f}')
