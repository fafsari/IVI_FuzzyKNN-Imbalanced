import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from SMOTEIVIFKNN import preprocess_data


def read_data(file_path):
    """Function to read data from an Excel file."""
    try:
        # Read Excel file into pandas dataframe
        data = pd.read_excel(file_path, engine='openpyxl')
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None


def k_fold_cross_validation(X, y, n_splits=5, n_neighbors=5):
    """Function to perform K-fold cross-validation."""
    
    # Initialize KFold and KNN model
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Variables to store results
    accuracies = []

    # K-fold Cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/{n_splits}")

        # Split data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the KNN model on the training data
        knn.fit(X_train, y_train)

        # Predict test data labels
        y_pred = knn.predict(X_test)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Accuracy for fold {fold}: {accuracy * 100:.2f}%")

    # Calculate the mean accuracy across all folds
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nMean accuracy across all folds: {mean_accuracy * 100:.2f}%")

def main(file_path):
    """Main function to execute the entire K-fold cross-validation process."""
    # Read the data
    data = read_data(file_path)

    # Preprocess the data
    X = data.drop(columns=['target'])  # Replace 'target' with your actual target column name
    y = data['target']  # Replace 'target' with your actual target column name

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply Interval-Valued Intuitionistic Fuzzy KNN (IVIF-KNN) based IPF for preprocessing
    X, y = preprocess_data(X, y)

    k_fold_cross_validation(X, y)

if __name__ == "__main__":
    file_path = 'your_data_file.xlsx'  # Replace with your file path
    main(file_path)
