Mood-Based Movie Recommendation System
A PyTorch-powered web app that recommends movies based on your current mood, detected through natural language input and mapped via the Arousal-Valence framework.

Features
Text input for describing your mood

Emotion classification using a PyTorch deep learning model

Mood-to-genre mapping based on the Arousal-Valence model

Diverse movie recommendations with rating filters

User-friendly interface built with Streamlit

Technologies Used
Python

Streamlit

PyTorch

Pandas & NumPy

Machine learning algorithms

Project Structure
text
.
├── src/        # Source code files
├── data/       # Movie datasets
├── docs/       # Documentation and reports
├── README.md   # You are here!
Installation & Setup
Clone the repository:

bash
git clone <repo_url>
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py
Usage
Launch the app

Enter a sentence describing your mood

Receive movie recommendations tailored to your mood
