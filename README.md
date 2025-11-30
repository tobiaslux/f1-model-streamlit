# F1 Team Win Predictor 🏎️

Machine learning model to predict F1 race winners based on weather and track conditions.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

That's it! The app will open automatically in your browser.

## Features
- Predicts win probability for all F1 teams
- Based on circuit, weather conditions, and starting grid position
- Trained on historical F1 race data

## Model
- Random Forest Classifier
- Features: Circuit, weather (temperature, precipitation, windspeed), grid position, year, round