Mood-Based Movie Recommendation System
Overview
This project is a mood-based movie recommendation system that suggests movies based on the user's current mood. The system uses natural language processing to analyze the user's input, classifies the emotion using a deep learning model, and then recommends movies aligned with the detected mood employing the Arousal-Valence framework.

Features
Text input for describing mood

Emotion classification using a PyTorch deep learning model

Mood-to-genre mapping based on Arousal-Valence model

Diverse movie recommendations with rating filters

User-friendly interface built with Streamlit

Technologies Used
Python

Streamlit

PyTorch

Pandas & NumPy

Machine Learning algorithms

Folder Structure
src/: Contains source code files

data/: Movie dataset and related data files

docs/: Documentation and reports

README.md: This file

Setup Instructions
Clone the repository:

text
git clone <repo_url>
Install dependencies:

text
pip install -r requirements.txt
Run the application:

text
streamlit run app.py
Usage
Launch the app and type a sentence describing your mood

The system predicts your mood and provides a list of recommended movies
