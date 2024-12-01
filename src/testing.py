
import numpy as np
from classification import classify_instance

# Function to perform Leave-One-Out Cross-Validation
def loocv(X, y, k, distance_metric):
    predictions = []
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        prediction = classify_instance(X_train, y_train, X_test[0], k, distance_metric)
        predictions.append((y_test, prediction))
    return predictions