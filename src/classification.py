from knn_implementation import get_neighbors, predict_classification

# Function to classify a single instance
def classify_instance(X_train, y_train, test_instance, k, distance_metric):
    neighbors = get_neighbors(X_train, y_train, test_instance, k, distance_metric)
    return predict_classification(neighbors)

# Function to classify a test set and log predictions
def classify_test_set(X_train, y_train, X_test, k, distance_metric):
    predictions = []
    for test_instance in X_test:
        neighbors = get_neighbors(X_train, y_train, test_instance, k, distance_metric)
        prediction = predict_classification(neighbors)
        predictions.append(prediction)
    return predictions