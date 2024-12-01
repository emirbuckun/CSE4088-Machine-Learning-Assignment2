import numpy as np

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    a = a.astype(int)
    b = b.astype(int)
    return np.sqrt(np.sum((a - b) ** 2))

# Function to calculate Manhattan distance
def manhattan_distance(a, b):
    a = a.astype(int)
    b = b.astype(int)
    return np.sum(np.abs(a - b))

# Function to get the k nearest neighbors
def get_neighbors(X_train, y_train, test_instance, k, distance_metric):
    distances = []
    for i in range(len(X_train)):
        if distance_metric == 'euclidean':
            dist = euclidean_distance(X_train[i], test_instance)
        elif distance_metric == 'manhattan':
            dist = manhattan_distance(X_train[i], test_instance)
        distances.append((y_train[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Function to make a prediction based on neighbors
def predict_classification(neighbors):
    return max(set(neighbors), key=neighbors.count)