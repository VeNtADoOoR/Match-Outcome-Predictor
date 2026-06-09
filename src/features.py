import pandas as pd

def calculate_rolling_form(df, window=5):
    """
    Calculates the points earned in the last 'window' matches for each team.
    """
    #Calculate the raw points earned in the current match
    # 1 = Home Win, 2 = Away Win, 0 = Draw
    df['home_points'] = df['outcome'].map({1: 3, 0: 1, 2: 0})
    df['away_points'] = df['outcome'].map({1: 0, 0: 1, 2: 3})

    #Build a unified chronological timeline for every individual club
    home_timeline = df[['game_id', 'date', 'home_club_id', 'home_points']].copy()
    home_timeline.rename(columns={'home_club_id': 'club_id', 'home_points': 'points'}, inplace=True)

    away_timeline = df[['game_id', 'date', 'away_club_id', 'away_points']].copy()
    away_timeline.rename(columns={'away_club_id': 'club_id', 'away_points': 'points'}, inplace=True)

    timeline = pd.concat([home_timeline, away_timeline])
    timeline = timeline.sort_values(by=['club_id', 'date'])

    #Calculate rolling totals but shift by 1 match
    timeline['form_points'] = timeline.groupby('club_id')['points'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
    )
    timeline['form_points'] = timeline['form_points'].fillna(0)

    #Map the calculated form back into the main match rows
    form_lookup = timeline[['game_id', 'club_id', 'form_points']]

    #Home Form mapping
    df = pd.merge(df, form_lookup, left_on=['game_id', 'home_club_id'], right_on=['game_id', 'club_id'], how='left')
    df.rename(columns={'form_points': 'home_form'}, inplace=True)
    df.drop(columns=['club_id'], inplace=True)

    # Away Form mapping
    df = pd.merge(df, form_lookup, left_on=['game_id', 'away_club_id'], right_on=['game_id', 'club_id'], how='left')
    df.rename(columns={'form_points': 'away_form'}, inplace=True)
    df.drop(columns=['club_id'], inplace=True)

    #Drop temporary variables and generate the relative advantage score
    df.drop(columns=['home_points', 'away_points'], inplace=True)
    df['form_difference'] = df['home_form'] - df['away_form']

    return df

def build_all_features(clean_games_df, clubs_df=None):
    """
    Master function to execute the feature engineering pipeline.
    Focuses strictly on dynamic temporal features.
    """
    df = clean_games_df.copy()
    
    #date conversion
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') 

    #Compute Rolling Performance Form
    df = calculate_rolling_form(df, window=5)
    
    return df