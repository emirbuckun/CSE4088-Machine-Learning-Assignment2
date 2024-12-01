# k-NN Classifier for Play Tennis Dataset

This project implements a custom k-Nearest Neighbors (k-NN) classifier for the Play Tennis dataset. It includes training, testing, evaluation, and visualization of results.

## Setup Instructions

1. **Clone the repository:**

```sh
   git clone https://github.com/emirbuckun/CSE4088-Machine-Learning-Assignment2.git
   cd CSE4088-Machine-Learning-Assignment2
```

2. **Create a virtual environment:**

```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies:**

```sh
pip install -r requirements.txt
```

4. **Prepare the dataset:**

- Ensure the data/play_tennis_data.json file is present in the data directory.

## Running the Program

1. **Run the main script:**

```sh
    python src/main.py
```

2. **Follow the prompts to enter the distance metric and the number of neighbors (k)::**

```sh
    Enter distance metric (euclidean/manhattan): euclidean
    Enter the number of neighbors (k): 3
```

3. **View the results::**

- The results will be saved in the results directory and printed to the console.

## Key Sections of The Code

**main.py**

- **Loading Data:** Loads the training set in JSON format and encodes using one-hot encoding.
- **User Input:** Takes user input for the distance metric and the number of neighbors (k).
- **Data Splitting:** Splits the dataset into training and testing sets.
- **Classification:** Classifies the test set using the custom k-NN classifier.
- **Evaluation:** Evaluates the accuracy of the model and logs the results.
- **Visualization:** Plots the confusion matrix and visualizes accuracy for different values of k.
- **LOOCV:** Performs Leave-One-Out Cross-Validation and evaluates the model.

**data_preparation.py**

- **Data Loading:** Implements functions to load and preprocess the dataset.
- **Encoding Categorical Data:** Implements one-hot encoding for categorical features.

**knn_implementation.py**

- **Distance Calculation:** Implements functions to calculate Euclidean and Manhattan distances.
- **Neighbor Selection:** Implements a function to get the k nearest neighbors based on the chosen distance metric.
- **Prediction:** Implements a function to make a prediction based on the neighbors.

**classification.py**

- **Instance Classification:** Implements a function to classify a single instance using the custom k-NN classifier.
- **Test Set Classification:** Implements a function to classify a test set and log predictions.

**evaluation.py**

- **Performance Evaluation:** Implements a function to evaluate the performance of the classifier using accuracy and confusion matrix.
