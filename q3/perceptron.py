"""
Logistic Regression (Perceptron) Implementation
This script implements a logistic regression model from scratch using NumPy 
to classify fruits (apples vs bananas) based on their physical characteristics.
"""

# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 2: Load and preprocess the data
def load_and_preprocess_data():
    data = pd.read_csv('fruit.csv')
    
    # Separate features and labels
    X = data[['length_cm', 'weight_g', 'yellow_score']].values
    y = data['label'].values
    
    # Feature scaling (normalize features)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    return X, y

# Step 3: Define the Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
    def forward(self, X):
        """Forward propagation"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute classification accuracy"""
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def backward(self, X, y_true, y_pred):
        """Compute gradients"""
        m = X.shape[0]
        dw = np.dot(X.T, (y_pred - y_true)) / m
        db = np.mean(y_pred - y_true)
        return dw, db
    
    def train(self, X, y, epochs=500, verbose=True):
        """Train the model"""
        n_features = X.shape[1]
        self.initialize_parameters(n_features)
        
        history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X)
            
            # Compute metrics
            loss = self.compute_loss(y, y_pred)
            accuracy = self.compute_accuracy(y, y_pred)
            
            # Backward propagation
            dw, db = self.backward(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store metrics
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
            # Early stopping if loss is small enough
            if loss < 0.05:
                print(f"Reached target loss at epoch {epoch}")
                break
                
        return history

def plot_training_history(history):
    """Plot the training progress"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss vs epoch
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    
    # Plot accuracy vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1)
    history = model.train(X, y)
    
    # Print final weights
    print("\nFinal model parameters:")
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    
    # Plot training progress
    plot_training_history(history)

if __name__ == "__main__":
    main() 