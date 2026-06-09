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

        # Check file exists (with a helpful printout of exactly where it looked)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: I looked exactly here -> {path}")

        df = pd.read_csv(path)

        # Basic validation
        print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

        dataframes[name] = df

    return dataframes


def validate_games(df):
    """
    Validate the games dataframe has required columns
    and no nulls in critical fields.
    """
    required_columns = [
        'game_id',
        'home_club_id',
        'away_club_id',
        'home_club_goals',
        'away_club_goals',
        'date',
        'season',
        'competition_id'
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Games table missing columns: {missing}")

    # Drop rows where goals are null — can't create target variable
    before = len(df)
    df = df.dropna(subset=['home_club_goals', 'away_club_goals'])
    after = len(df)

    if before != after:
        print(f"Dropped {before - after} rows with missing goals")

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
            return 'H'
        elif row['home_club_goals'] < row['away_club_goals']:
            return 'A'
        else:
            return 'D'

    df['outcome'] = df.apply(get_outcome, axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts(normalize=True).round(3))

    return df