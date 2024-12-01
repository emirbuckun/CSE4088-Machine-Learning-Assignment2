import numpy as np

def evaluate_performance(y_true, y_pred):
    # Convert 'Yes'/'No' to 1/0 if necessary
    y_true = np.array([1 if val == 'Yes' else 0 for val in y_true], dtype=int)
    y_pred = np.array([1 if val == 'Yes' else 0 for val in y_pred], dtype=int)
    
    # Calculate accuracy
    accuracy = (np.sum(y_true == y_pred) / len(y_true)) * 100
    
    # Initialize confusion matrix components
    tp = fp = tn = fn = 0
    
    # Calculate confusion matrix components
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn += 1
    
    return accuracy, tn, fp, fn, tp