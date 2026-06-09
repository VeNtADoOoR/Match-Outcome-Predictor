import pandas as pd
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "Data")

def load_data():
    """
    Load all relevant tables and perform basic validation.
    Returns a dictionary of dataframes.
    """
    files = {
        'games': 'games.csv',
        'clubs': 'clubs.csv',
    }

    dataframes = {}

    for name, filename in files.items():
        # Safely join the dynamic root path with the file name
        path = os.path.join(DATA_PATH, filename)

        # Check file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: -> {path}")

        df = pd.read_csv(path)

        # Basic validation
        print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

        dataframes[name] = df

    return dataframes

def clean_data(df):
    """
    Performs deep cleaning: removes duplicates and restricts data to recent seasons.
    """
    initial_rows = df.shape[0]
    
    df = df.drop_duplicates().copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        #I kept the data from 2019 onward
        df = df[df['date'].dt.year >= 2019]
    
    # 3. Final index reset
    df = df.reset_index(drop=True)
    
    rows_dropped = initial_rows - df.shape[0]
    print(f"Deep cleaning complete. Dropped {rows_dropped} dirty/outdated rows. {df.shape[0]} rows remaining.")
    return df

def validate_games(df):
    """
    Cleans the games dataset by dropping irrelevant columns and missing scores.
    """
    initial_rows = df.shape[0]
    initial_cols = df.shape[1]
    
    # Columns I think they are irrelevant for this prediction (I might be wrong)
    columns_to_drop = [
        'home_club_position', 'away_club_position', 
        'attendance', 'stadium', 'referee',
        'home_club_manager_name', 'away_club_manager_name', 
        'home_club_formation', 'away_club_formation',
        'home_club_name', 'away_club_name','url', 'aggregate', 'round'
    ]
    
    df = df.drop(columns=columns_to_drop, errors='ignore').copy()
    df = df.dropna(subset=['home_club_goals', 'away_club_goals'])
    
    print(f"Games Validation: Dropped {len(columns_to_drop)} messy columns.")
    print(f"Remaining shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    data = load_data()
    games = validate_games(data['games'])
    print(f"\nGames ready for processing: {games.shape[0]} matches")

def create_target(df):
    """
    Create match outcome column.
    H = Home win, D = Draw, A = Away win
    """
    def get_outcome(row):
        if row['home_club_goals'] > row['away_club_goals']:
            return 1
        elif row['home_club_goals'] < row['away_club_goals']:
            return 2
        else:
            return 0

    df['outcome'] = df.apply(get_outcome, axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts(normalize=True).round(3))

    return df