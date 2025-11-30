import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Load model
@st.cache_resource
def load_model():
    return joblib.load('f1_team_winner_model.pkl')

model = load_model()

# Active teams
ACTIVE_TEAMS = [
    {'id': 131, 'name': 'Mercedes'},
    {'id': 6, 'name': 'Ferrari'},
    {'id': 9, 'name': 'Red Bull'},
    {'id': 3, 'name': 'McLaren'},
    {'id': 214, 'name': 'Aston Martin'},
    {'id': 4, 'name': 'Alpine F1 Team'},
    {'id': 51, 'name': 'Alfa Romeo'},
    {'id': 213, 'name': 'Haas F1 Team'},
    {'id': 5, 'name': 'Williams'},
    {'id': 210, 'name': 'AlphaTauri'}
]

CIRCUITS = {
    24: "Abu Dhabi Grand Prix",
    1: "Australian Grand Prix",
    70: "Austrian Grand Prix",
    73: "Azerbaijan Grand Prix",
    3: "Bahrain Grand Prix",
    13: "Belgian Grand Prix",
    18: "Brazilian Grand Prix",
    9: "British Grand Prix",
    7: "Canadian Grand Prix",
    17: "Chinese Grand Prix",
    21: "Emilia Romagna Grand Prix",
    34: "French Grand Prix",
    11: "Hungarian Grand Prix",
    14: "Italian Grand Prix",
    22: "Japanese Grand Prix",
    80: "Las Vegas Grand Prix",
    32: "Mexican Grand Prix",
    79: "Miami Grand Prix",
    6: "Monaco Grand Prix",
    78: "Qatar Grand Prix",
    77: "Saudi Arabian Grand Prix",
    15: "Singapore Grand Prix",
    4: "Spanish Grand Prix",
    69: "United States Grand Prix"
}

# Page config
st.set_page_config(page_title="F1 Team Win Predictor", page_icon="🏎️", layout="wide")

# Title
st.title("🏎️ F1 Team Win Predictor")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Race Parameters")
    
    circuit_name = st.selectbox("Circuit", list(CIRCUITS.values()))
    circuit_id = [k for k, v in CIRCUITS.items() if v == circuit_name][0]
    
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
    round_num = st.number_input("Round", min_value=1, max_value=24, value=1)
    
    st.subheader("Weather Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    windspeed = st.number_input("Windspeed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    
    predict_button = st.button("🏁 Predict Winner", type="primary", use_container_width=True)

with col2:
    st.subheader("Predictions")
    
    if predict_button:
        with st.spinner("Analyzing race conditions..."):
            # Predict for all teams
            predictions = []
            
            for team in ACTIVE_TEAMS:
                input_df = pd.DataFrame({
                    'year_x': [year],
                    'round_x': [round_num],
                    'temperature': [temperature],
                    'precipitation': [precipitation],
                    'windspeed': [windspeed],
                    'circuitId': [circuit_id],
                    'constructorId': [team['id']]
                })
                
                prob = model.predict_proba(input_df)[0, 1]
                predictions.append({
                    'Team': team['name'],
                    'Win Probability': f"{prob * 100:.1f}%",
                    'prob_value': prob
                })
            
            # Sort by probability
            predictions.sort(key=lambda x: x['prob_value'], reverse=True)

            # Display top 3
            st.markdown(f"### 🏆 Top 3 Predictions for {circuit_name}")
            
            for i, pred in enumerate(predictions[:3]):
                if i == 0:
                    emoji = "🥇"
                    color = "gold"
                elif i == 1:
                    emoji = "🥈"
                    color = "silver"
                else:
                    emoji = "🥉"
                    color = "#cd7f32"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(225, 6, 0, 0.2), rgba(0, 0, 0, 0.8)); 
                            padding: 20px; border-radius: 10px; margin: 10px 0; 
                            border-left: 5px solid {color};">
                    <h3 style="margin: 0; color: {color};">{emoji} {i+1}. {pred['Team']}</h3>
                    <h2 style="margin: 5px 0; color: #e10600;">{pred['Win Probability']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all predictions in table
            st.markdown("### 📈 All Teams")
            df = pd.DataFrame(predictions)[['Team', 'Win Probability']]
            st.dataframe(df, use_container_width=True, hide_index=True)

            predictions.sort(key=lambda x: x['prob_value'], reverse=True)
            st.markdown("### 📊 Win Probabilities")
            prob_df = pd.DataFrame(predictions)[['Team', 'prob_value']]

            # Matplotlib figure
            plt.style.use('dark_background')

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(
                prob_df['Team'], 
                prob_df['prob_value'], 
                color='#e10600'  # F1 Red
            )

            ax.set_xlabel("Win Probability")
            ax.set_ylabel("Team")
            ax.set_xlim(0, max(prob_df['prob_value']) * 1.1)

            # Add % labels
            for i, v in enumerate(prob_df['prob_value']):
                ax.text(v + 0.005, i, f"{v*100:.1f}%", va='center', color='white')

            plt.gca().invert_yaxis()  # Highest on top

            st.pyplot(fig)
            # ---------------------------------
    else:
        st.info("👆 Fill in the race parameters and click 'Predict Winner' to see predictions")

# Footer
st.markdown("---")
st.markdown("*Predictions based on historical F1 race data and weather conditions*")