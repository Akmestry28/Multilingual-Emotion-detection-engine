import streamlit as st, torch
from models.base import MultilingualEmotionDetector
from utils.relationship_emotion_styles import get_relationship_guidance
from utils.seed import set_seed
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


langs = ['hi','mr','gu','pa','bn','ta','ur','en']
rels = ['mother','father','friend','sibling']

# -----------------------------
# Seed & Device Setup
# -----------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultilingualEmotionDetector(device=device)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Indian Multilingual Emotion Detection Engine")
text = st.text_area("Enter text")
lang = st.selectbox("Language", langs)
rel = st.selectbox("Relationship", rels)

if st.button("Detect Emotion"):
    if not text:
        st.warning("Please enter some text to analyze.")
    else:
        preds = model.predict_emotions([text])[0]
        preds_sorted = sorted(preds, key=lambda x: x['probability'], reverse=True)

        dominant = preds_sorted[0]['emotion']
        intensity = preds_sorted[0]['intensity']

        # Removed cultural expression call (was using get_cultural_expression)
        guidance_text, tone, support_level = get_relationship_guidance(rel, dominant, lang)

        st.json({
            "detected_emotions": preds_sorted,
            "dominant_emotion": dominant,
            "emotion_intensity": round(intensity, 4),
            "response_tone": tone,
            "relationship_guidance": guidance_text,
            "emotional_support_level": support_level
        })
