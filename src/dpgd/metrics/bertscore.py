"""BERTScore metric wrapper."""

from typing import List, Optional


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "es",
    model_type: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Compute BERTScore for predictions against references.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        lang: Language code (default: "es" for Spanish)
        model_type: Optional BERT model type
        device: Device to run on (e.g., "cuda", "cpu")
        
    Returns:
        Dictionary with 'precision', 'recall', 'f1' scores (as lists)
    """
    try:
        from bert_score import score
    except ImportError:
        raise ImportError(
            "bert-score not installed. Install with: pip install bert-score"
        )
    
    # Use default model for Spanish if not specified
    if model_type is None:
        if lang == "es":
            model_type = "dccuchile/bert-base-spanish-wwm-uncased"
        else:
            model_type = "bert-base-uncased"
    
    def _call_score(num_layers: Optional[int] = None):
        return score(
            predictions,
            references,
            lang=lang,
            model_type=model_type,
            device=device,
            verbose=False,
            num_layers=num_layers,
        )

    try:
        P, R, F1 = _call_score()
    except KeyError as err:
        # Some custom Hugging Face checkpoints are not in bert-score's internal
        # lookup table (model2layers). When that happens we try to infer the
        # number of layers directly from the model config and retry.
        if err.args and err.args[0] == model_type:
            num_layers = _infer_num_layers(model_type)
            if num_layers is None:
                raise RuntimeError(
                    f"Unable to infer the number of layers for '{model_type}'. "
                    "Please specify a bert-score compatible model or set "
                    "`model_type` explicitly."
                ) from err
            P, R, F1 = _call_score(num_layers=num_layers)
        else:
            raise

    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "precision_mean": float(P.mean()),
        "recall_mean": float(R.mean()),
        "f1_mean": float(F1.mean()),
    }


def _infer_num_layers(model_name: str) -> Optional[int]:
    """Infer the number of transformer layers for a model from its config."""
    try:
        from transformers import AutoConfig
    except ImportError:  # pragma: no cover - transformers is already a dependency
        return None

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        return None

    for attr in (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "encoder_layers",
        "layers",
    ):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return int(value)
    return None

