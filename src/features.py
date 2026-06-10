import pandas as pd

def calculate_dynamic_features(df, window=5):
    """
    Rips the dataset into a single timeline to calculate form, goal difference, 
    and rest days, then merges them back as relative advantages.
    """
    #Map Match Points
    df['home_points'] = df['outcome'].map({1: 3, 0: 1, 2: 0})
    df['away_points'] = df['outcome'].map({1: 0, 0: 1, 2: 3})

    #Build the unified timeline
    home_timeline = df[['game_id', 'date', 'home_club_id', 'home_points', 'home_club_goals', 'away_club_goals', 'season']].copy()
    home_timeline.rename(columns={
        'home_club_id': 'club_id', 'home_points': 'points', 
        'home_club_goals': 'goals_scored', 'away_club_goals': 'goals_conceded'
    }, inplace=True)

    away_timeline = df[['game_id', 'date', 'away_club_id', 'away_points', 'away_club_goals', 'home_club_goals', 'season']].copy()
    away_timeline.rename(columns={
        'away_club_id': 'club_id', 'away_points': 'points', 
        'away_club_goals': 'goals_scored', 'home_club_goals': 'goals_conceded'
    }, inplace=True)

    timeline = pd.concat([home_timeline, away_timeline])
    timeline = timeline.sort_values(by=['club_id', 'date'])
    
    #Rolling Form
    timeline['form_points'] = timeline.groupby(['club_id', 'season'])['points'].transform(
        lambda x: x.rolling(window, min_periods=1).sum().shift(1)).fillna(0)
        
    #Rolling Goal Difference
    timeline['rolling_scored'] = timeline.groupby(['club_id', 'season'])['goals_scored'].transform(
        lambda x: x.rolling(window, min_periods=1).sum().shift(1)).fillna(0)
    timeline['rolling_conceded'] = timeline.groupby(['club_id', 'season'])['goals_conceded'].transform(
        lambda x: x.rolling(window, min_periods=1).sum().shift(1)).fillna(0)
    timeline['rolling_gd'] = timeline['rolling_scored'] - timeline['rolling_conceded']

    #Rest Days
    timeline['prev_match_date'] = timeline.groupby('club_id')['date'].shift(1)
    timeline['rest_days'] = (timeline['date'] - timeline['prev_match_date']).dt.days
    # Assuming 7 days in case it's their first match in the table
    timeline['rest_days'] = timeline['rest_days'].fillna(7)

    #Map everything back to the main match rows
    lookup = timeline[['game_id', 'club_id', 'form_points', 'rolling_gd', 'rest_days']]

    # Home Mapping
    df = pd.merge(df, lookup, left_on=['game_id', 'home_club_id'], right_on=['game_id', 'club_id'], how='left')
    df.rename(columns={'form_points': 'home_form', 'rolling_gd': 'home_gd', 'rest_days': 'home_rest'}, inplace=True)
    df.drop(columns=['club_id'], inplace=True)

    # Away Mapping
    df = pd.merge(df, lookup, left_on=['game_id', 'away_club_id'], right_on=['game_id', 'club_id'], how='left')
    df.rename(columns={'form_points': 'away_form', 'rolling_gd': 'away_gd', 'rest_days': 'away_rest'}, inplace=True)
    df.drop(columns=['club_id'], inplace=True)

    # 5. Create the Final Relative Advantages!
    df.drop(columns=['home_points', 'away_points'], inplace=True)
    df['form_difference'] = df['home_form'] - df['away_form']
    df['goal_difference_advantage'] = df['home_gd'] - df['away_gd']
    df['rest_days_advantage'] = df['home_rest'] - df['away_rest']

    return df

def build_all_features(clean_games_df, clubs_df=None):
    """
    Master function to execute the temporal feature engineering pipeline.
    """
    df = clean_games_df.copy()
    
    # Force date conversion and chronological order
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') 
    
    # Optional: Bring back readable club names for debugging
    if clubs_df is not None:
        df = pd.merge(df, clubs_df[['club_id', 'name']], left_on='home_club_id', right_on='club_id', how='left')
        df.rename(columns={'name': 'home_club_name'}).drop(columns=['club_id'], errors='ignore', inplace=True)
        df = pd.merge(df, clubs_df[['club_id', 'name']], left_on='away_club_id', right_on='club_id', how='left')
        df.rename(columns={'name': 'away_club_name'}).drop(columns=['club_id'], errors='ignore', inplace=True)

    # Execute the timeline engine
    df = calculate_dynamic_features(df, window=5)
    
    return df