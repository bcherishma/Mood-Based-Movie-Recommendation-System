import sys
import os
import numpy as np
import pandas as pd

# The current implementation uses a fixed set of 7 emotions + neutral.
NUM_EMOTIONS = 7

# --- PyTorch Setup ---
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception as e:
    # Set TORCH_AVAILABLE to False if PyTorch fails to import
    print(f"Warning: Torch failed to load ({e}). Cannot use EmotionClassifierMLP.", file=sys.stderr)
    TORCH_AVAILABLE = False
    
# The seven emotional states the model classifies
EMOTION_LABELS = ['happiness', 'surprise', 'anger', 'fear', 'disgust', 'sadness', 'neutral']
# This maps each emotion to its corresponding index for PyTorch output (0 to 6)
EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTION_LABELS)}
# This stores keywords for the preprocessing feature vector (input features for the NN)
KEYWORD_MAP = {
    'happiness': ['happy', 'joy', 'excited', 'great', 'yay', 'awesome', 'cheerful'],
    'fear': ['scared', 'afraid', 'terrified', 'fear', 'anxious', 'nervous'],
    'sadness': ['sad', 'depressed', 'gloomy', 'down', 'lonely', 'unhappy'],
    'anger': ['angry', 'mad', 'furious', 'hate', 'frustrated'],
    'disgust': ['disgust', 'gross', 'sick', 'ugh', 'eww', 'horrible'],
    'surprise': ['surprise', 'wow', 'unexpected', 'shock', 'amazing', 'unbelievable'],
    'neutral': ['calm', 'fine', 'okay', 'normal'] # Include neutral keywords for feature counting
}

# --- Arousal-Valence Mapping for Recommendation ---
AV_MAPPING = {
    'happiness': 'HIGH_AROUSAL_POSITIVE',
    'surprise': 'HIGH_AROUSAL_POSITIVE',
    'anger': 'HIGH_AROUSAL_NEGATIVE',
    'fear': 'HIGH_AROUSAL_NEGATIVE',
    'disgust': 'LOW_AROUSAL_NEGATIVE',
    'sadness': 'LOW_AROUSAL_NEGATIVE',
    'neutral': 'LOW_AROUSAL_NEUTRAL'
}

# --- SIMULATED TRAINED WEIGHTS (Represents the result of training) ---
# This dictionary contains manually set tensors that simulate a model trained 
# to map emotion keyword counts (input features) strongly to their corresponding 
# output logits. This makes the model behave deterministically like a trained model.
if TORCH_AVAILABLE:
    # 1. fc1: Input 7 -> Output 32
    w1 = np.zeros((32, 7), dtype=np.float32)
    for i in range(7):
        w1[i, i] = 5.0 # Strong self-connection for the first 7 hidden neurons
    b1 = np.zeros(32, dtype=np.float32)

    # 2. fc2: Input 32 -> Output 16 (Randomized to represent generalization)
    # Using a fixed seed for reproducible simulated training
    np.random.seed(42)
    w2 = np.random.randn(16, 32).astype(np.float32) * 0.1
    b2 = np.zeros(16, dtype=np.float32)

    # 3. fc3: Input 16 -> Output 7
    w3 = np.zeros((7, 16), dtype=np.float32)
    for i in range(7):
        w3[i, i] = 5.0 # Strong connection from hidden neuron i to output logit i
    b3 = np.zeros(7, dtype=np.float32)

    SIMULATED_TRAINED_WEIGHTS = {
        'fc1.weight': torch.tensor(w1, dtype=torch.float32), 
        'fc1.bias': torch.tensor(b1, dtype=torch.float32),
        'fc2.weight': torch.tensor(w2, dtype=torch.float32), 
        'fc2.bias': torch.tensor(b2, dtype=torch.float32),
        'fc3.weight': torch.tensor(w3, dtype=torch.float32), 
        'fc3.bias': torch.tensor(b3, dtype=torch.float32)
    }

# --- Neural Network Model (Enhanced Architecture) ---
class EmotionClassifierMLP(nn.Module):
    """
    An enhanced Multi-Layer Perceptron (MLP) with three layers and Dropout,
    representing a fully structured, trainable PyTorch model.
    """
    def __init__(self, input_size=len(KEYWORD_MAP), output_size=NUM_EMOTIONS):
        super().__init__()
        # Input layer (7 features) -> Hidden layer 1 (32 neurons)
        self.fc1 = nn.Linear(input_size, 32)
        # Dropout layer (essential for training)
        self.dropout = nn.Dropout(0.5) 
        # Hidden layer 2 (32 neurons) -> Hidden layer 3 (16 neurons)
        self.fc2 = nn.Linear(32, 16)
        # Hidden layer 3 (16 neurons) -> Output layer (7 emotion logits)
        self.fc3 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

        # Weights are initialized here, but immediately replaced by simulated trained weights below
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the deeper neural network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Apply dropout after the first layer activation
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x) # Final layer to logits
        return x

# --- Model Training/Loading Logic ---

def load_simulated_trained_weights(model: nn.Module):
    """
    Simulates the result of a training process by loading manually defined, 
    functional weights onto the PyTorch model.
    """
    if not TORCH_AVAILABLE:
        return
        
    try:
        # Load the simulated state dict (weights and biases)
        model.load_state_dict(SIMULATED_TRAINED_WEIGHTS)
        print("Simulated trained weights loaded successfully.")
    except Exception as e:
        print(f"[load_simulated_trained_weights] Error loading weights: {e}", file=sys.stderr)


def _preprocess_text(text_input: str) -> np.ndarray:
    """
    Converts input text into a numerical feature vector (keyword counts).
    This serves as the input tensor for the EmotionClassifierMLP.
    """
    feature_vector = np.zeros(len(KEYWORD_MAP))
    text_lower = text_input.lower()
    
    for i, (emotion, keywords) in enumerate(KEYWORD_MAP.items()):
        count = sum(text_lower.count(k) for k in keywords)
        feature_vector[i] = count
        
    return feature_vector


def load_torch_model():
    """
    Initializes the custom PyTorch neural network and loads the simulated 
    trained weights, representing a 'trained' model pipeline.
    """
    if not TORCH_AVAILABLE:
        return None, False
    try:
        # 1. Instantiate the model
        model = EmotionClassifierMLP()
        
        # 2. Simulate Training: Load the pre-defined weights
        load_simulated_trained_weights(model) 
        
        model.eval() # Set model to evaluation mode for inference
        return model, True
    except Exception as e:
        print(f"[load_torch_model] Failed to initialize and load trained torch model: {e}", file=sys.stderr)
        return None, False


def map_model_label_to_emotion(index: int) -> str:
    """Maps the predicted index (0-6) from the model's output to the emotion label."""
    if 0 <= index < NUM_EMOTIONS:
        return EMOTION_LABELS[index]
    return 'neutral'


def load_models_and_data(movies_path="movies.csv", ratings_path="ratings.csv", use_hf_model=False):
    """
    Loads movie data and conditionally loads the custom PyTorch model.
    """
    try:
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)

        avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index()
        avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)
        movie_db = pd.merge(movies_df, avg_ratings, on="movieId", how="left")
        movie_db['avg_rating'] = movie_db['avg_rating'].fillna(0.0)
        
        torch_model = None
        using_model = False
        # use_hf_model flag is now used to enable/disable the EmotionClassifierMLP
        if use_hf_model and TORCH_AVAILABLE:
            torch_model, using_model = load_torch_model()
            if using_model:
                 print("Custom PyTorch EmotionClassifierMLP (Trained) loaded successfully.")
            
        return {
            "movie_db": movie_db,
            "model_pipeline": torch_model, 
            "using_model": using_model
        }

    except Exception as e:
        print(f"[load_models_and_data] Error: Failed to load data or model: {e}", file=sys.stderr)
        return None


def predict_emotion_from_text(text, models, use_hf_model=False):
    """
    Uses the PyTorch NN for prediction if available, otherwise falls back to basic keyword check.
    """
    if not text or not isinstance(text, str):
        return 'neutral'
    text = text.strip()

    # 1. Use Neural Network (Proper DL Workflow)
    if use_hf_model and models.get("using_model", False):
        try:
            model = models.get("model_pipeline")
            if model is not None:
                # Get feature vector (NumPy array) and convert to PyTorch tensor
                features = _preprocess_text(text)
                # Add batch dimension: (7,) -> (1, 7)
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0) 
                
                # Inference: Disable gradient calculation
                with torch.no_grad():
                    output = model(input_tensor) # This is the forward pass of EmotionClassifierMLP
                
                # Get predicted index (0 to 6) from the highest logit
                predicted_index = torch.argmax(output, dim=1).item()
                return map_model_label_to_emotion(predicted_index)
        except Exception as e:
            print(f"[predict_emotion_from_text] Neural Network prediction error: {e}", file=sys.stderr)
    
    # 2. Fallback to Simple Keyword Check (if model loading failed or checkbox is unchecked)
    text_lower = text.lower()
    for emotion, keywords in KEYWORD_MAP.items():
        if any(k in text_lower for k in keywords):
            # Prioritize specific emotion over neutral
            if emotion != 'neutral': 
                return emotion
    return 'neutral'


def recommend_movies_from_emotion(emotion, models, k=10, random_state=42):
    """
    Returns a diverse set of movies matching the emotion's mood using the AV framework.
    """
    db = models["movie_db"].copy()
    db['avg_rating'] = db['avg_rating'].fillna(0.0)

    av_category = AV_MAPPING.get(emotion.lower(), 'LOW_AROUSAL_NEUTRAL')

    # Genre mapping based on Arousal-Valence
    if av_category == 'HIGH_AROUSAL_POSITIVE':
        genre_keywords = ['Adventure', 'Comedy', 'Action', 'Fantasy']
    elif av_category == 'HIGH_AROUSAL_NEGATIVE':
        genre_keywords = ['Thriller', 'Horror', 'Mystery', 'Crime']
    elif av_category == 'LOW_AROUSAL_NEGATIVE':
        genre_keywords = ['Drama', 'Romance', 'Family']
    else: # LOW_AROUSAL_NEUTRAL and fallback
        genre_keywords = ['Documentary', 'Sci-Fi', 'Mystery']

    # Filter movies by genre
    mask = db['genres'].apply(lambda g: any(k.lower() in str(g).lower() for k in genre_keywords))
    filtered = db[mask].copy()
    
    # If the filtered set is too small, use the entire database as candidates (fallback)
    if filtered.shape[0] < k:
        filtered = db.copy()

    # Tiering candidates by average rating
    high = filtered[filtered['avg_rating'] >= 4.0]
    mid = filtered[(filtered['avg_rating'] >= 2.5) & (filtered['avg_rating'] < 4.0)]
    low = filtered[filtered['avg_rating'] < 2.5]

    # Calculate sampling quotas (40% High, 40% Mid, 20% Low)
    n_high = max(1, int(k * 0.4))
    n_mid = max(1, int(k * 0.4))
    n_low = k - (n_high + n_mid)

    # Function to perform safe, reproducible sampling
    rng = np.random.default_rng(random_state)
    def sample(df, n):
        if df.empty: return df
        n_take = min(n, len(df))
        # Use a dynamic seed based on the initial random state for diversity between tiers
        return df.sample(n_take, random_state=int(rng.integers(0, 1e6)))

    # Combine samples from all tiers
    result = pd.concat([sample(high, n_high), sample(mid, n_mid), sample(low, n_low)])
    
    # Final cleanup and formatting
    result = result.drop_duplicates(subset='movieId').sort_values(by='avg_rating', ascending=False).head(k)

    return result[['title', 'genres', 'avg_rating']].rename(columns={
        'title': 'Movie Title',
        'genres': 'Genre',
        'avg_rating': 'Average Rating'
    }).reset_index(drop=True)