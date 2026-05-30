PROMPTS = {

    # ========================================================
    # GENERAL BASELINE (LLM ONLY)
    # ========================================================
    "generic_ml": """
You are a machine learning expert.

Assign importance scores (0–1) to features based on predictive power.

Return ONLY JSON:
{{"feature": score}}

Features:
{features}
""",

    # ========================================================
    # BIOMEDICAL GENERAL
    # ========================================================
    "biomedical_general": """
You are an expert biomedical AI system.

TASK:
Assign importance scores (0–1) to biomedical features.

Focus on:
- biological relevance
- clinical relevance
- predictive power

Return ONLY valid JSON only.

Features:
{features}
""",

    # ========================================================
    # CANCER DOMAIN
    # ========================================================
    "cancer_clinical": """
You are a clinical AI expert specializing in cancer diagnosis.

Evaluate gene expression features based on:
- oncological relevance
- biomarker importance
- redundancy reduction
- predictive power

Return ONLY JSON:
{{"feature": score}}

Features:
{features}
""",

    # ========================================================
    # PARKINSON DOMAIN
    # ========================================================
    "parkinson_clinical": """
You are a clinical AI expert specializing in neurological disorders.

Evaluate features for Parkinson's disease prediction using:
- motor symptom relevance
- neurological biomarkers
- signal stability
- predictive relevance

Return ONLY JSON:
{{"feature": score}}

Features:
{features}
""",

    # ========================================================
    # STRICT FORMAT (MOST STABLE FOR LLM)
    # ========================================================
    "strict_json": """
You are a biomedical AI system.

Return ONLY valid JSON with:
- keys = feature names
- values = float in [0,1]

No text. No explanation. No markdown.

Features:
{features}
"""
}