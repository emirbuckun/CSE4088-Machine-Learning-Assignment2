import pandas as pd
import json

def create_dataset():
    # Manually create the "Play Tennis" dataset
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        "PlayTennis": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    }
    
    # Save the dataset as a JSON file
    with open('data/play_tennis_data.json', 'w') as f:
        json.dump(data, f)
    print("Dataset created and saved to 'data/play_tennis_data.json'")

def preprocess_data(input_path, output_path):
    # Load the raw data
    print("Loading raw data from:", input_path)
    df = pd.read_json(input_path)
    print("Raw data loaded successfully.")
    print("Raw data:\n", df.head())

    # Print summary of the dataset
    print("\nSummary of the dataset:")
    print(df.describe(include='all'))
    print("\nNumber of instances per class:")
    print(df['PlayTennis'].value_counts())

    # Encode categorical variables
    print("Encoding categorical variables...")
    df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
    print("Categorical variables encoded successfully.")
    print("Encoded data:\n", df_encoded.head())

    # Save the encoded data to a JSON file
    print("Saving encoded data to:", output_path)
    df_encoded.to_json(output_path, orient='records')
    print("Encoded data saved successfully.")