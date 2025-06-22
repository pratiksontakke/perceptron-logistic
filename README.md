# 🍎 Apple vs Banana Classifier 🍌

## Project Overview
This project implements a logistic regression model (single-neuron perceptron) from scratch using NumPy to classify fruits (apples vs bananas) based on their physical characteristics.

## 📊 Dataset
The model uses `fruit.csv` containing measurements of 12 fruits:
- 6 apples (label: 0)
- 6 bananas (label: 1)

### Features:
- `length_cm`: Length of the fruit in centimeters
- `weight_g`: Weight of the fruit in grams
- `yellow_score`: Yellowness score (0-1, where 1 is very yellow)

## 💻 Implementation Details
- Pure NumPy implementation
- Batch gradient descent
- Binary cross-entropy loss
- Early stopping when loss < 0.05

## 📈 Results

### Training Performance
- Initial Loss: 0.6869
- Convergence: Reached target loss at epoch 80
- Final Accuracy: 100%

### Model Parameters
```python
Weights:
- length_cm:    1.10577622  (longer → more likely banana)
- weight_g:    -0.94315059  (heavier → more likely apple)
- yellow_score: 1.09601754  (yellower → more likely banana)

Bias: -0.010550934406717657
```

### Interpretation
The model learned these key patterns:
1. Bananas are characterized by:
   - Greater length
   - Higher yellow score
   - Lower weight

2. Apples are characterized by:
   - Shorter length
   - Lower yellow score
   - Higher weight

## 📁 Project Structure
```
q3/
├── perceptron.py      # Main implementation
├── fruit.csv          # Dataset
└── README.md          # This file
```

## 🔍 Key Findings
1. **Perfect Classification**: The model achieved 100% accuracy on the training data
2. **Quick Convergence**: Reached optimal performance in just 80 epochs
3. **Intuitive Weights**: The learned parameters align with real-world fruit characteristics

### Feature Importance
1. Length (1.106): Most important feature for identifying bananas
2. Yellow Score (1.096): Strong indicator for bananas
3. Weight (-0.943): Key feature for identifying apples (negative correlation with banana class)

## 🚀 Running the Project
1. Install requirements:
```bash
pip install numpy pandas matplotlib
```

2. Run the classifier:
```bash
python perceptron.py
```

## 📊 Visualization
The training process generates plots showing:
- Loss vs Epoch
- Accuracy vs Epoch

These visualizations demonstrate the model's learning progress and convergence.

## 🎯 Learning Outcomes
This implementation demonstrates:
1. Basic neural network concepts
2. Gradient descent optimization
3. Binary classification
4. Feature normalization
5. Model convergence monitoring 