import streamlit as st
import pandas as pd
import joblib
import os
import sys
from xgboost import XGBClassifier
from src.ingest import load_data, validate_games, create_target, clean_data
from src.features import build_all_features

st.set_page_config(page_title="Match outcome predictor", page_icon="⚽", layout="centered")
st.title("⚽ Match outcome predictor")
st.text("By VeNtADoOoR", text_alignment = 'center')
st.markdown("Filter by league, then select two teams to fetch their actual latest stats.")
st.divider()

LEAGUE_NAMES = {
    'GB1': 'Premier League (England)',
    'ES1': 'La Liga (Spain)',
    'IT1': 'Serie A (Italy)',
    'L1':  'Bundesliga (Germany)',
    'FR1': 'Ligue 1 (France)',
    'PO1': 'Primeira Liga (Portugal)',
    'NL1': 'Eredivisie (Netherlands)',
    'BE1': 'Jupiler Pro League (Belgium)',
    'TR1': 'Süper Lig (Turkey)',
    'GR1': 'Super League 1 (Greece)',
    'RU1': 'Premier Liga (Russia)',
    'UKR1' :'Premier Liga (Ukraine)',
    'DK1': 'Superligaen (Denmark)',
    'SC1': 'Scottish Premiership (Scotland)'
}

@st.cache_data
def load_and_prep_database():
    data = load_data()
    games = clean_data(create_target(validate_games(data['games'])))
    final_df = build_all_features(games, data['clubs'])
    return final_df, data['clubs']

@st.cache_resource 
def get_or_train_model(_df):
    if os.path.exists('../xgboost_engine.pkl'):
        return joblib.load('../xgboost_engine.pkl')
    elif os.path.exists('xgboost_engine.pkl'):
        return joblib.load('xgboost_engine.pkl')
    
    features = ['form_difference', 'goal_difference_advantage', 'rest_days_advantage']
    ml_df = _df.dropna(subset=features).copy()
    X = ml_df[features]
    y = ml_df['outcome']
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='mlogloss')
    model.fit(X, y)
    
    joblib.dump(model, 'xgboost_engine.pkl')
    return model

with st.spinner("Connecting to data ..."):
    df, clubs_df = load_and_prep_database()
    model = get_or_train_model(df)
    
    valid_club_ids = set(df['home_club_id']).union(set(df['away_club_id']))
    active_clubs = clubs_df[clubs_df['club_id'].isin(valid_club_ids)].copy()

if 'domestic_competition_id' in active_clubs.columns:
    available_leagues = active_clubs['domestic_competition_id'].dropna().unique()
    available_leagues = sorted(list(available_leagues))
    st.subheader("🌍 Select Competition")
    selected_league_id = st.selectbox(
        "League", 
        options=available_leagues,
        format_func=lambda x: LEAGUE_NAMES.get(x, x),
        label_visibility="collapsed"
    )
    filtered_clubs = active_clubs[active_clubs['domestic_competition_id'] == selected_league_id]
else:
    filtered_clubs = active_clubs

club_dict = dict(zip(filtered_clubs['name'], filtered_clubs['club_id']))
team_names = sorted(list(club_dict.keys()))

def get_real_team_stats(club_id, dataframe):
    club_matches = dataframe[(dataframe['home_club_id'] == club_id) | (dataframe['away_club_id'] == club_id)].copy()
    if club_matches.empty:
        return {'form': 0, 'gd': 0, 'rest_days': 7}
        
    last_match = club_matches.sort_values('date', ascending=False).iloc[0]
    
    if last_match['home_club_id'] == club_id:
        return {'form': last_match['home_form'], 'gd': last_match['home_gd'], 'rest_days': last_match['home_rest']}
    else:
        return {'form': last_match['away_form'], 'gd': last_match['away_gd'], 'rest_days': last_match['away_rest']}

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select Home Team:", team_names, index=0)
    
with col2:
    st.subheader("✈️ Away Team")
    safe_index = 1 if len(team_names) > 1 else 0
    away_team = st.selectbox("Select Away Team:", team_names, index=safe_index)

if home_team == away_team:
    st.warning("⚠️ Please select two different teams.")
else:
    home_id = club_dict[home_team]
    away_id = club_dict[away_team]

    home_stats = get_real_team_stats(home_id, df)
    away_stats = get_real_team_stats(away_id, df)
    
    form_diff = home_stats['form'] - away_stats['form']
    gd_adv = home_stats['gd'] - away_stats['gd']
    rest_adv = home_stats['rest_days'] - away_stats['rest_days']

    expected_features = model.feature_names_in_
    input_dict = {
        'form_difference': [form_diff],
        'goal_difference_advantage': [gd_adv],
        'rest_days_advantage': [rest_adv]
    }

    input_data = pd.DataFrame(input_dict)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Match", type="primary", use_container_width=True):
        
        probs = model.predict_proba(input_data)[0]
        
        with st.expander("🔍 Features"):
            st.write(f"**{home_team}** (Last Known Form): {home_stats['form']} pts | Goal Diff: {home_stats['gd']} | Rested: {home_stats['rest_days']} days")
            st.write(f"**{away_team}** (Last Known Form): {away_stats['form']} pts | Goal Diff: {away_stats['gd']} | Rested: {away_stats['rest_days']} days")
            st.write(f"**Net Advantage for {home_team}:** Form ({form_diff}), GD ({gd_adv}), Rest ({rest_adv})")
            if 'h2h_dominance' in input_dict:
                st.write(f"**Historical H2H Dominance:** {input_dict['h2h_dominance'][0]:.2f}")

        st.divider()
        st.subheader(f"📊 The model's prediction: {home_team} vs {away_team}")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(f"🏠 {home_team} Win", f"{probs[1] * 100:.1f}%")
        res_col2.metric("🤝 Draw", f"{probs[0] * 100:.1f}%")
        res_col3.metric(f"✈️ {away_team} Win", f"{probs[2] * 100:.1f}%")
        
        st.progress(float(probs[1]), text="Home Win Probability")
        st.progress(float(probs[0]), text="Draw Probability")
        st.progress(float(probs[2]), text="Away Win Probability")