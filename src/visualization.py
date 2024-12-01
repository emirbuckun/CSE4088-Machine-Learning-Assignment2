import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from testing import loocv
from evaluation import evaluate_performance

# Function to plot confusion matrix
def plot_confusion_matrix(tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Function to visualize accuracy for different values of k
def plot_accuracy_vs_k(X, y, distance_metric, max_k=10):
    accuracies = []
    ks = range(1, max_k + 1)
    for k in ks:
        loocv_predictions = loocv(X, y, k, distance_metric)
        y_true = [true for true, pred in loocv_predictions]
        y_pred = [pred for true, pred in loocv_predictions]
        accuracy, _, _, _, _ = evaluate_performance(np.array(y_true), np.array(y_pred))
        accuracies.append(accuracy)
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. k')
    plt.show()