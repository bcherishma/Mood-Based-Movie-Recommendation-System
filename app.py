import streamlit as st
import pandas as pd
import main  

st.set_page_config(page_title="NLP Mood-Based Movie Recommender", layout="wide")
st.title("Mood-Based Movie Recommender")
st.markdown(
    "Type a sentence describing your current mood. "
    "The app detects your emotion and recommends movies based on your mood"
)
st.markdown("---")

st.sidebar.header("Model Configuration")
use_dl_model = st.sidebar.checkbox(
    "Use Deep Learning (Torch) Model", 
    value=True, 
    help="If checked, uses the custom PyTorch Dummy Model for classification. If unchecked, defaults to simple keyword matching."
)

@st.cache_resource
def get_cached_data(use_dl_model_flag: bool):
    """
    Cache the result of main.load_models_and_data(use_hf_model=use_dl_model_flag)
    so model & movie DB load only once while Streamlit session runs.
    """
    return main.load_models_and_data(use_hf_model=use_dl_model_flag)

models = get_cached_data(use_dl_model)

if not models:
    st.error(
        "Failed to load models or data. Check that `movies.csv` and `ratings.csv` exist "
        "in the working directory and check the Streamlit console for details."
    )
    st.stop()

using_model = bool(models.get("using_model"))
model_status = "Custom PyTorch Model LOADED (Keyword-Based) " if using_model else "PyTorch Failed to Load (using keyword fallback) ⚠️"
st.info(f"Model status: {model_status}")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Describe Your Mood")
    user_mood_text = st.text_input(
        "What kind of day are you having? (e.g., 'I feel so joyful and excited!')",
        placeholder="Type your mood here...",
        key="mood_input"
    )

    st.subheader("Recommendation Settings")
    k = st.slider("How many recommendations to show (k)", min_value=3, max_value=20, value=10, step=1)
    randomize = st.checkbox("Randomize results each request (no fixed seed)", value=False)
    recommend_button = st.button("Get Recommendation")

with col2:
    st.header("2. Your Movie Recommendation")
    emotion_placeholder = st.empty()
    recommendation_placeholder = st.empty()
    meta_placeholder = st.empty()  
if recommend_button:
    if not user_mood_text:
        emotion_placeholder.warning("Please enter a sentence describing your mood before requesting recommendations.")
    else:
        predicted_emotion = main.predict_emotion_from_text(user_mood_text, models, use_hf_model=use_dl_model)
        emotion_placeholder.info(f"**Predicted Emotion:** **{predicted_emotion.upper()}**")

        seed = None if randomize else 42

        try:
            recommendations = main.recommend_movies_from_emotion(predicted_emotion, models, k=k, random_state=seed)
        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {e}")
            recommendations = None

        if isinstance(recommendations, pd.DataFrame):
            recommendation_placeholder.subheader(f"Top {k} recommendations for **{predicted_emotion}**:")
            recommendation_placeholder.dataframe(recommendations)
            
            try:
                predicted_emotion_lower = predicted_emotion.lower()
                av_cat = main.AV_MAPPING.get(predicted_emotion_lower, "LOW_AROUSAL_NEUTRAL")
                
                if av_cat == 'HIGH_AROUSAL_POSITIVE':
                    genre_keywords = ['Adventure', 'Comedy', 'Action', 'Fantasy']
                elif av_cat == 'HIGH_AROUSAL_NEGATIVE':
                    genre_keywords = ['Thriller', 'Horror', 'Mystery', 'Crime']
                elif av_cat == 'LOW_AROUSAL_NEGATIVE':
                    genre_keywords = ['Drama', 'Romance', 'Family']
                else: 
                    genre_keywords = ['Documentary', 'Sci-Fi', 'Mystery']

                movie_db = models.get("movie_db")
                if movie_db is not None and 'genres' in movie_db.columns:
                    candidate_mask = movie_db['genres'].apply(
                        lambda g: any(k.lower() in str(g).lower() for k in genre_keywords)
                    )
                    n_candidates = int(candidate_mask.sum())
                    
                    genre_list = ", ".join(genre_keywords)
                    meta_placeholder.caption(f"AV Category: **{av_cat}** · Target Genres: **{genre_list}** · Candidates Found: **{n_candidates}**")
            except Exception as e:
                meta_placeholder.caption(f"Metadata error: {e}")
        else:
            recommendation_placeholder.error("No recommendations available. Check console for errors.")

if 'recommendation_given' not in st.session_state:
    if not recommend_button:
        st.session_state['recommendation_given'] = False
        recommendation_placeholder.dataframe(
            pd.DataFrame({
                'Movie Title': ['Waiting for your mood...'],
                'Genre': ['Waiting for your mood...'],
                'Average Rating': ['Waiting for your mood...']
            })
        )

if recommend_button:
    st.session_state['recommendation_given'] = True
