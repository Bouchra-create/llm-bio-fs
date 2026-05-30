import json
import requests
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from src.prompts import PROMPTS

LLM_URL = "http://172.24.100.5:11434/api/generate"


# ============================================================
# SAFE JSON PARSER (MORE ROBUST)
# ============================================================

def clean_json(text):

    try:
        # remove code blocks if any
        text = text.replace("```json", "").replace("```", "")

        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end <= start:
            return None

        json_text = text[start:end]

        return json.loads(json_text)

    except Exception as e:
        print("JSON PARSE ERROR:", e)
        return None


# ============================================================
# LLM FEATURE SCORING
# ============================================================

def get_llm_scores(
    feature_names,
    model="qwen2.5-coder:7b",
    prompt_name="generic_ml",
    max_retries=2
):

    prompt_template = PROMPTS[prompt_name]
    scores = {}

    batch_size = 10

    feature_names = list(feature_names)

    for i in range(0, len(feature_names), batch_size):

        batch = feature_names[i:i + batch_size]

        prompt = prompt_template.format(features=list(batch))

        response_text = None
        data = None

        # ====================================================
        # RETRY LOGIC (IMPORTANT FOR STABILITY)
        # ====================================================
        for attempt in range(max_retries):

            try:

                response = requests.post(
                    LLM_URL,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0,
                            "num_predict": 300,
                            "seed": 42
                        }
                    },
                    timeout=120
                )

                response.raise_for_status()
                response_text = response.json().get("response", "")

                data = clean_json(response_text)

                if data is not None:
                    break

            except Exception as e:
                print(f"LLM ERROR (attempt {attempt+1}):", e)

        # ====================================================
        # FALLBACK STRATEGY
        # ====================================================
        if data is None:

            print("INVALID JSON RESPONSE → fallback uniform score")

            for f in batch:
                scores[f] = 0.5

            continue

        # ====================================================
        # STORE SCORES
        # ====================================================
        for k, v in data.items():

            try:
                scores[k] = float(v)
            except:
                scores[k] = 0.5

    # ============================================================
    # FILL MISSING FEATURES
    # ============================================================
    for f in feature_names:
        if f not in scores:
            scores[f] = 0.5

    # ============================================================
    # NORMALIZATION (RESEARCH SAFE)
    # ============================================================
    vec = np.array([scores[f] for f in feature_names], dtype=float)

    vec = np.nan_to_num(vec, nan=0.5, posinf=1.0, neginf=0.0)

    if np.max(vec) > np.min(vec):
        vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
    else:
        vec = np.ones_like(vec) * 0.5

    return vec, scores