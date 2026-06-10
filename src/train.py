from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from ingest import load_data, validate_games, create_target, clean_data
from features import build_all_features

def prepare_training_data(df):
    """
    Slices the master dataset into the exact features the model is allowed to see.
    """
    features = [
        'form_difference', 
        'goal_difference_advantage', 
        'rest_days_advantage'
    ]
    
    # Drop rows that have NaN values
    ml_df = df.dropna(subset=features).copy()
    
    X = ml_df[features]
    y = ml_df['outcome']
    
    return X, y

def train_baseline_model(X_train, y_train):
    """
    Initializes and trains the Random Forest classifier.
    """
    # n_estimators = 100 decision trees voting on the outcome
    # max_depth = 5 prevents the model from over-complicating and memorizing the data
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Tests the model on unseen data and prints the final report.
    """
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print("\n" + "="*40)
    print("*|* BASELINE MODEL RESULTS *|*")
    print("="*40)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    # 0 = Draw, 1 = Home Win, 2 = Away Win
    print(classification_report(y_test, predictions, target_names=['Draw (0)', 'Home Win (1)', 'Away Win (2)']))

if __name__ == "__main__":
    #Run the entire ETL pipeline automatically
    print("Loading and cleaning data...")
    data = load_data()
    games = validate_games(data['games'])
    games = create_target(games)
    clean_games = clean_data(games)
    
    print("Calculating dynamic features...")
    final_data = build_all_features(clean_games, data['clubs'])
    
    X, y = prepare_training_data(final_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_baseline_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)