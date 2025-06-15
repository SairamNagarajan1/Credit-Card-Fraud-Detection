# fraud_detection.py

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

# Step 2: Load Data
data = pd.read_csv("creditcard.csv")
print("Credit card data received")
print(data.head())

# Step 3: Describe Dataset
print("\nDataset Description:")
print(data.describe())

# Step 4: Analyze Class Distribution
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))

print(f"\nOutlier Fraction: {outlier_fraction:.6f}")
print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Transactions: {len(valid)}")

# Step 5: Explore Transaction Amounts
print("\nAmount details of the fraudulent transactions:")
print(fraud.Amount.describe())

print("\nAmount details of valid transactions:")
print(valid.Amount.describe())

# Step 6: Correlation Heatmap
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 7: Prepare Data
X = data.drop(['Class'], axis=1)
Y = data['Class']

xData = X.values
yData = Y.values

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42
)

# Step 8: Train Model
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Step 9: Predict
yPred = rfc.predict(xTest)

# Step 10: Evaluate
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Step 11: Confusion Matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

