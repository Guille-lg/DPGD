"""Generation wrapper for DPGD decoding."""

from typing import Dict, List, Optional, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    GenerationConfig,
)

from ..embeddings.difficulty_embeddings import DifficultyEmbeddingSet
from ..profiles.profile_schema import HybridDifficultyProfile
from .dpgd_logits_processor import DPGDLogitsProcessor


def generate_with_dpgd(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    profile: HybridDifficultyProfile,
    embedding_set: DifficultyEmbeddingSet,
    input_text: Union[str, List[str]],
    alpha_SDP: float = 1.0,
    alpha_Freq: float = 1.0,
    alpha_Morph: float = 1.0,
    delta_SDP: float = 0.5,
    lambda_strength: float = 1.0,
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    num_beams: int = 1,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **kwargs,
) -> Union[str, List[str]]:
    """
    Generate text using DPGD (Dynamic Profile-Guided Decoding).
    
    Args:
        model: HuggingFace model for generation
        tokenizer: HuggingFace tokenizer
        profile: HybridDifficultyProfile with difficulty scores
        embedding_set: DifficultyEmbeddingSet with embeddings
        input_text: Input text or list of input texts
        alpha_SDP: Weight for Semantic Distance Penalty
        alpha_Freq: Weight for frequency penalty
        alpha_Morph: Weight for morphology penalty
        delta_SDP: Threshold/margin for SDP calculation. Must be > 0 for SDP
            to have any effect. Paper specifies δ ∈ (0, 2]. Default: 0.5
        lambda_strength: Global personalization strength (scales all penalties)
        max_length: Maximum total sequence length
        max_new_tokens: Maximum number of new tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        pad_token_id: Padding token ID (uses tokenizer default if None)
        eos_token_id: End-of-sequence token ID (uses tokenizer default if None)
        **kwargs: Additional arguments passed to model.generate()
        
    Returns:
        Generated text(s) as string or list of strings
    """
    # Handle single string input
    is_single = isinstance(input_text, str)
    if is_single:
        input_text = [input_text]
    
    # Tokenize inputs
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set default token IDs
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # Create DPGD logits processor
    logits_processor = DPGDLogitsProcessor(
        profile=profile,
        embedding_set=embedding_set,
        tokenizer=tokenizer,
        model=model,
        alpha_SDP=alpha_SDP,
        alpha_Freq=alpha_Freq,
        alpha_Morph=alpha_Morph,
        delta_SDP=delta_SDP,
        lambda_strength=lambda_strength,
        device=device,
    )
    
    # Prepare generation config
    generation_config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=[logits_processor],
            **kwargs,
        )
    
    # Decode outputs
    # Remove input tokens from output
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]
    
    generated_texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    # Return single string or list
    if is_single:
        return generated_texts[0]
    return generated_texts


def generate_simplified(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    profile: HybridDifficultyProfile,
    embedding_set: DifficultyEmbeddingSet,
    source_text: str,
    alpha_SDP: float = 1.0,
    alpha_Freq: float = 1.0,
    alpha_Morph: float = 1.0,
    delta_SDP: float = 0.5,
    lambda_strength: float = 1.0,
    **generation_kwargs,
) -> str:
    """
    Simplified interface for generating simplified text.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        profile: HybridDifficultyProfile
        embedding_set: DifficultyEmbeddingSet
        source_text: Source text to simplify
        alpha_SDP: Weight for Semantic Distance Penalty
        alpha_Freq: Weight for frequency penalty
        alpha_Morph: Weight for morphology penalty
        delta_SDP: Threshold/margin for SDP calculation. Must be > 0 for SDP
            to have any effect. Paper specifies δ ∈ (0, 2]. Default: 0.5
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Simplified text
    """
    return generate_with_dpgd(
        model=model,
        tokenizer=tokenizer,
        profile=profile,
        embedding_set=embedding_set,
        input_text=source_text,
        alpha_SDP=alpha_SDP,
        alpha_Freq=alpha_Freq,
        alpha_Morph=alpha_Morph,
        delta_SDP=delta_SDP,
        lambda_strength=lambda_strength,
        **generation_kwargs,
    )

