# models/base.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultilingualEmotionDetector:
    """
    Two modes:
      - mode='zero_shot'  -> uses multilingual XLM-R NLI zero-shot pipeline (no training required)
      - mode='encoder'    -> encoder + linear head (requires fine-tuning)
    """
    def __init__(self, emotions=None, device=None, mode='zero_shot',
                 encoder_model_name="xlm-roberta-base", classifier_checkpoint=None):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[INFO] Using device: {self.device}")
        self.emotions = emotions or ['joy','sadness','anger','fear','love','surprise','excitement']
        self.mode = mode
        self.classifier_checkpoint = classifier_checkpoint

        if self.mode == 'zero_shot':
            from transformers import pipeline
            nli_model = "joeddav/xlm-roberta-large-xnli"
            device_idx = 0 if self.device.type == "cuda" else -1
            self.zs_clf = pipeline("zero-shot-classification", model=nli_model, device=device_idx)
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
            print("[INFO] zero-shot pipeline ready (joeddav/xlm-roberta-large-xnli).")
        else:
            base_name = self.classifier_checkpoint if self.classifier_checkpoint else encoder_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(base_name)
            self.base_model = AutoModel.from_pretrained(base_name)
            hidden_dim = self.base_model.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, len(self.emotions) * 2)
            )
            self.base_model.to(self.device)
            self.classifier.to(self.device)
            self.base_model.eval()
            self.classifier.eval()
            print(f"[INFO] Loaded encoder ({base_name}) + classifier head. "
                  f"IMPORTANT: fine-tune before use for correct outputs.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def predict_emotions(self, texts):
        """
        Returns: list (per input) of dicts {'emotion','probability','intensity'}
        """
        if self.mode == 'zero_shot':
            results = []
            for t in texts:
                out = self.zs_clf(t, candidate_labels=self.emotions, multi_label=True)
                label_to_score = {lab: scr for lab, scr in zip(out['labels'], out['scores'])}
                preds = [
                    {
                        'emotion': e,
                        'probability': float(label_to_score.get(e, 0.0)),
                        'intensity': float(label_to_score.get(e, 0.0))
                    }
                    for e in self.emotions
                ]
                results.append(preds)
            return results

        else:
            self.base_model.eval()
            self.classifier.eval()
            encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.base_model(input_ids=encodings['input_ids'],
                                          attention_mask=encodings['attention_mask'])
                pooled = self._mean_pooling(outputs, encodings['attention_mask'])
                logits = self.classifier(pooled)
                scores = torch.sigmoid(logits)
            results = []
            for s in scores:
                probs = s[:len(self.emotions)].cpu().tolist()
                intensities = s[len(self.emotions):].cpu().tolist()
                preds = [
                    {'emotion': self.emotions[i], 'probability': probs[i], 'intensity': intensities[i]}
                    for i in range(len(self.emotions))
                ]
                results.append(preds)
            return results
