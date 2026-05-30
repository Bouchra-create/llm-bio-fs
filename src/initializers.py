import numpy as np
from src.llm import get_llm_scores
from src.filters import compute_filter_scores

def get_init_scores(init_type, X, y, features, model=None, prompt=None, filter_type=None):

    if init_type == "llm":
        scores, _ = get_llm_scores(
            features,
            model=model,
            prompt_name=prompt
        )
        return scores

    elif init_type == "filter":
        return compute_filter_scores(
            X, y,
            method=filter_type
        )

    elif init_type == "random":
        return np.random.rand(len(features))

    else:
        raise ValueError("Unknown init type")