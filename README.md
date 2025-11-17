# 🎬 Mood-Based Movie Recommendation System

A PyTorch-powered web application that recommends movies based on your current mood detected through natural language input and mapped using the Arousal–Valence model.

---

## ⭐ Overview

Mood-Based Movie Recommendation System analyzes how you feel through a text description,  
classifies your emotion using a deep learning model,  
converts it into Arousal–Valence coordinates,  
and recommends movies that match your emotional state.

It provides an interactive and user-friendly interface built with Streamlit.

---

## 🚀 Features

- **Natural Language Mood Input** – Describe how you feel in your own words.  
- **Emotion Classification (PyTorch Model)** – Detects emotional states from text.  
- **Arousal–Valence Mapping** – Converts emotions to psychological dimensions.  
- **Mood-to-Genre Mapping** – Recommends movies that best match your mood.  
- **Rating Filters** – Apply IMDb rating thresholds.  
- **Interactive Streamlit UI** – Fast, clean, and intuitive interface.  

---

## 🧠 Technologies Used

- Python  
- Streamlit  
- PyTorch  
- Pandas  
- NumPy  
- Machine Learning Algorithms  

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.x  
- pip package manager  

### Steps

**1. Clone the repository**
```bash
git clone <repo_url>
cd <project_folder>
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. Launch the Streamlit web app.  
2. Enter a short sentence describing your current mood.  
3. The system predicts your emotion → maps arousal & valence → finds relevant genres.  
4. Receive movie recommendations tailored to your mood.  

---

