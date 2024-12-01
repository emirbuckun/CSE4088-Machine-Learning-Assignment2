import pandas as pd
import numpy as np
import os

from data_preparation import create_dataset, preprocess_data
from classification import classify_test_set
from evaluation import evaluate_performance
from testing import loocv
from visualization import plot_accuracy_vs_k, plot_confusion_matrix
from knn_implementation import get_neighbors

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Check if the raw data file exists
raw_data_path = 'data/play_tennis_data.json'
if not os.path.exists(raw_data_path):
    # Create the dataset if the raw data file does not exist
    print("Raw data file not found. Creating dataset...")
    create_dataset()

# Check if the encoded data file exists
encoded_data_path = 'data/encoded_play_tennis_data.json'
if not os.path.exists(encoded_data_path):
    # Preprocess the data if the encoded file does not exist
    print("Encoded data file not found. Preprocessing raw data...")
    preprocess_data(raw_data_path, encoded_data_path)
    print("Data preprocessing completed. Encoded data saved to:", encoded_data_path)

# Load the training set in JSON format
df_encoded = pd.read_json(encoded_data_path)

# Split the dataset into features and target variable
X = df_encoded.drop('PlayTennis', axis=1).values
y = df_encoded['PlayTennis'].values

# Take user input for distance metric and k
distance_metric = input("Enter distance metric (euclidean/manhattan): ").strip().lower()
k = int(input("Enter the number of neighbors (k): ").strip())

# Validate k
if k <= 0:
    raise ValueError("k must be a positive integer")

# Split the dataset into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Classify the test set and log predictions
predictions = classify_test_set(X_train, y_train, X_test, k, distance_metric)

# Evaluate the accuracy of the model
accuracy, tn, fp, fn, tp = evaluate_performance(y_test, predictions)

# Save results to a file
result_log = open('results/test_set_results.txt', 'w')
result_log.write("\nAccuracy of custom k-NN classifier: {:.2f}%\n".format(accuracy))
result_log.write("Confusion Matrix:\n")
result_log.write(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}\n")
result_log.close()

# Print to console
print("\nAccuracy of custom k-NN classifier: {:.2f}%".format(accuracy))
print("Confusion Matrix:")
print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")

# Plot the confusion matrix
plot_confusion_matrix(tn, fp, fn, tp)

# Visualize accuracy for different values of k
plot_accuracy_vs_k(X, y, distance_metric, max_k=10)

# Log detailed outputs
detailed_log = open('results/detailed_test_set_results.txt', 'w')
for i, test_instance in enumerate(X_test):
    neighbors = get_neighbors(X_train, y_train, test_instance, k, distance_metric)
    detailed_log.write(f"\nTest Instance {i+1}:\n")
    detailed_log.write(f"Predicted: {predictions[i]}, Actual: {y_test[i]}\n")
    detailed_log.write(f"Neighbors: {neighbors}\n")
    print(f"\nTest Instance {i+1}:")
    print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
    print(f"Neighbors: {neighbors}")
detailed_log.close()

# Perform Leave-One-Out Cross-Validation
loocv_predictions = loocv(X, y, k, distance_metric)

# Evaluate the accuracy of the model using LOOCV
y_true = [true for true, pred in loocv_predictions]
y_pred = [pred for true, pred in loocv_predictions]
accuracy, tn, fp, fn, tp = evaluate_performance(np.array(y_true), np.array(y_pred))
loocv_log = open('results/loocv_results.txt', 'w')
loocv_log.write("\nLOOCV Accuracy of custom k-NN classifier: {:.2f}%\n".format(accuracy))
loocv_log.write("LOOCV Confusion Matrix:\n")
loocv_log.write(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}\n")
loocv_log.close()

# Print to console
print("\nLOOCV Accuracy of custom k-NN classifier: {:.2f}%".format(accuracy))
print("LOOCV Confusion Matrix:")
print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")

# Log detailed outputs for LOOCV
detailed_loocv_log = open('results/detailed_loocv_results.txt', 'w')
for i, (true, pred) in enumerate(loocv_predictions):
    detailed_loocv_log.write(f"\nLOOCV Test Instance {i+1}:\n")
    detailed_loocv_log.write(f"Predicted: {pred}, Actual: {true}\n")
    print(f"\nLOOCV Test Instance {i+1}:")
    print(f"Predicted: {pred}, Actual: {true}")
detailed_loocv_log.close()