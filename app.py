import argparse, torch, json
from models.base import MultilingualEmotionDetector
from utils.relationship_emotion_styles import get_relationship_guidance
from utils.seed import set_seed

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Multilingual Emotion Detection")
parser.add_argument("--text", help="Input text for emotion detection")
parser.add_argument(
    "--lang",
    choices=['hi','mr','gu','pa','bn','ta','ur','en'],
    help="Language code"
)
parser.add_argument(
    "--relationship",
    default="friend",
    choices=['mother','father','friend','sibling'],
    help="Relationship context"
)
args = parser.parse_args()

# -----------------------------
# Handle Missing Args (Defaults)
# -----------------------------
if not args.text or not args.lang:
    print("⚠️  No --text or --lang provided. Using defaults for demo.")
    if not args.text:
        args.text = "I am feeling very happy today!"
    if not args.lang:
        args.lang = "en"

# -----------------------------
# Seed & Device Setup
# -----------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultilingualEmotionDetector(device=device)

# -----------------------------
# ✅ FIX: Ensure tokenizer has pad_token
# -----------------------------
if hasattr(model, "tokenizer"):
    if model.tokenizer.pad_token is None:
        # Option 1: Use EOS token for padding (works for GPT-like models)
        if model.tokenizer.eos_token:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        else:
            # Option 2: Add a new [PAD] token if EOS doesn't exist
            model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(model, "model"):
                model.model.resize_token_embeddings(len(model.tokenizer))

# -----------------------------
# Prediction
# -----------------------------
preds = model.predict_emotions([args.text])[0]
preds_sorted = sorted(preds, key=lambda x: x['probability'], reverse=True)

dominant = preds_sorted[0]['emotion']
intensity = preds_sorted[0]['intensity']

# -----------------------------
# Relationship Mapping
# -----------------------------
guidance_text, tone, support_level = get_relationship_guidance(
    args.relationship, dominant, args.lang
)

# -----------------------------
# Output Formatting
# -----------------------------
output = {
    "input_text": args.text,
    "language": args.lang,
    "detected_emotions": [
        {
            "emotion": e['emotion'],
            "probability": f"{e['probability']*100:.2f}%"  # show as %
        }
        for e in preds_sorted
    ],
    "dominant_emotion": dominant,
    "emotion_intensity": round(intensity, 4),
    "response_tone": tone,
    "relationship_guidance": guidance_text,
    "emotional_support_level": support_level
}

print(json.dumps(output, ensure_ascii=False, indent=2))
