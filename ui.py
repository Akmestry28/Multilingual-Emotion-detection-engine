import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import streamlit as st

# -----------------------------
# MUST be the first Streamlit command
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection Engine",
    page_icon="💭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import torch
from models.base import MultilingualEmotionDetector
from utils.relationship_emotion_styles import get_relationship_guidance
from utils.seed import set_seed
import warnings
import plotly.express as px
import pandas as pd

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Define language and relationship options with display names
LANGUAGES = {
    'en': '🇺🇸 English',
    'hi': '🇮🇳 हिंदी (Hindi)',
    'mr': '🇮🇳 मराठी (Marathi)',
    'gu': '🇮🇳 ગુજરાતી (Gujarati)',
    'pa': '🇮🇳 ਪੰਜਾਬੀ (Punjabi)',
    'bn': '🇮🇳 বাংলা (Bengali)',
    'ta': '🇮🇳 தமிழ் (Tamil)',
    'ur': '🇵🇰 اردو (Urdu)'
}

RELATIONSHIPS = {
    'mother': '👩‍👧‍👦 Mother',
    'father': '👨‍👧‍👦 Father',
    'friend': '👫 Friend',
    'sibling': '👫 Sibling'
}

# -----------------------------
# Seed & Device Setup
# -----------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model in session state to avoid reloading
@st.cache_resource
def load_model():
    return MultilingualEmotionDetector(device=device, mode='zero_shot')

model = load_model()

# -----------------------------
# Custom CSS for better styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .subheader {
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .guidance-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Main UI
# -----------------------------
st.markdown('<h1 class="main-header">💭 Indian Multilingual Emotion Detection Engine</h1>', unsafe_allow_html=True)

st.markdown("### 📝 Enter your text for emotion analysis")

# Input section with columns
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "Text to analyze",
        placeholder="Type your message here... (supports multiple Indian languages)",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("#### ⚙️ Settings")
    
    # Language selection
    lang_display = st.selectbox(
        "🌐 Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=0
    )
    
    # Relationship selection
    rel_display = st.selectbox(
        "👥 Relationship Context",
        options=list(RELATIONSHIPS.keys()),
        format_func=lambda x: RELATIONSHIPS[x],
        index=0
    )

# Analysis button
st.markdown("---")
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
with analyze_col2:
    analyze_button = st.button(
        "🔍 Analyze Emotions",
        type="primary",
        use_container_width=True
    )

# Results section
if analyze_button:
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤔 Analyzing emotions..."):
            try:
                # Get predictions
                preds = model.predict_emotions([text])[0]
                preds_sorted = sorted(preds, key=lambda x: x['probability'], reverse=True)
                
                dominant = preds_sorted[0]['emotion']
                intensity = preds_sorted[0]['intensity']
                
                guidance_text, tone, support_level = get_relationship_guidance(
                    rel_display, dominant, lang_display
                )
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="subheader">📊 Analysis Results</h2>', unsafe_allow_html=True)
                
                # Main emotion result
                st.markdown(f'''
                <div class="emotion-card">
                    <h3>🎯 Dominant Emotion: {dominant.title()}</h3>
                    <h4>Intensity: {intensity:.2%}</h4>
                </div>
                ''', unsafe_allow_html=True)
                
                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎭 Emotion", dominant.title())
                with col2:
                    st.metric("📈 Intensity", f"{intensity:.1%}")
                with col3:
                    st.metric("🗣️ Response Tone", tone.title())
                
                # Relationship guidance
                st.markdown(f'''
                <div class="guidance-card">
                    <h3>💡 Relationship Guidance</h3>
                    <p><strong>Context:</strong> {RELATIONSHIPS[rel_display]}</p>
                    <p><strong>Support Level:</strong> {support_level}</p>
                    <p><strong>Guidance:</strong> {guidance_text}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Detailed emotions table
                with st.expander("📋 Detailed Emotion Breakdown"):
                    df = pd.DataFrame(preds_sorted)
                    df['emotion'] = df['emotion'].str.title()
                    df['probability'] = df['probability'].round(4)
                    df['intensity'] = df['intensity'].round(4)
                    st.dataframe(
                        df,
                        column_config={
                            "emotion": "Emotion",
                            "probability": st.column_config.ProgressColumn(
                                "Confidence",
                                help="Confidence score for this emotion",
                                min_value=0,
                                max_value=1,
                            ),
                            "intensity": "Intensity"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🚀 Powered by Advanced NLP | Supports 8 Indian Languages"
    "</div>", 
    unsafe_allow_html=True
)
