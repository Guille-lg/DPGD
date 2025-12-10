"""Experiment runner for DPGD evaluation."""

import json
import string
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from ..config import ConfigLoader
from ..data.alexsis_dataset import ALEXSISDataset
from ..decoding.generation import generate_with_dpgd
from ..embeddings.difficulty_embeddings import build_difficulty_embeddings
from ..metrics.bertscore import compute_bertscore
from ..metrics.profile_hit_rate import compute_phr, compute_phr_details
from ..metrics.readability_es import compute_szigriszt_pazos
from ..metrics.sari import compute_sari
from ..profiles.profile_builder import ProfileBuilder


def run_evaluation(
    config_path: str | Path,
    output_dir: str | Path = "./results",
    device: Optional[str] = None,
) -> Dict:
    """
    Run full evaluation pipeline.
    
    Steps:
    1. Load config and Data
    2. Build Profile and Embeddings
    3. Initialize Model and DPGDLogitsProcessor
    4. Run generation on test set
    5. Compute SARI, BERTScore, Readability, and PHR
    6. Save results to JSON/CSV
    
    Args:
        config_path: Path to experiment configuration file
        output_dir: Directory to save results
        device: Device to run on (e.g., "cuda", "cpu")
        
    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("DPGD Evaluation Pipeline")
    print("=" * 80)
    
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[STEP 1/6] Loading Configuration")
    print(f"  Config file: {config_path}")
    print(f"  Output directory: {output_dir}")
    
    # Step 1: Load config
    loader = ConfigLoader()
    config = loader.load(config_path)
    print(f"  ✓ Configuration loaded successfully")
    
    # Extract configuration
    model_config = config.get("model", {})
    profile_config = config.get("profile", {})
    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    
    model_name = model_config.get("name", "gpt2")
    model_path = model_config.get("path", model_name)
    
    # Profile parameters
    beta_Freq = profile_config.get("beta_Freq", 0.4)
    beta_Morph = profile_config.get("beta_Morph", 0.4)
    beta_Mask = profile_config.get("beta_Mask", 0.2)
    threshold = profile_config.get("threshold", 0.5)
    frequency_file = profile_config.get("frequency_file")
    long_word_threshold = profile_config.get("long_word_threshold", 12)
    
    # DPGD parameters
    alpha_SDP = profile_config.get("alpha_SDP", 1.0)
    alpha_Freq = profile_config.get("alpha_Freq", 1.0)
    alpha_Morph = profile_config.get("alpha_Morph", 1.0)
    delta_SDP = profile_config.get("delta_SDP", 0.5)  # Must be > 0 for SDP to work
    
    # Generation parameters
    max_new_tokens = experiment_config.get("max_new_tokens", 100)
    num_beams = experiment_config.get("num_beams", 1)
    do_sample = experiment_config.get("do_sample", True)
    temperature = experiment_config.get("temperature", 0.7)
    
    # Data parameters
    data_path = data_config.get("path", "data/alexsis.jsonl")
    test_split = data_config.get("test_split", 1.0)  # Use all data if no split
    
    print(f"\n[STEP 2/6] Loading Dataset")
    print(f"  Data path: {data_path}")
    print(f"  Test split: {test_split * 100:.1f}%")
    print(f"  Model: {model_name} ({model_path})")
    print(f"  Profile parameters: β_Freq={beta_Freq}, β_Morph={beta_Morph}, β_Mask={beta_Mask}")
    print(f"  DPGD parameters: α_SDP={alpha_SDP}, α_Freq={alpha_Freq}, α_Morph={alpha_Morph}, δ_SDP={delta_SDP}")
    print(f"  Generation: max_tokens={max_new_tokens}, beams={num_beams}, sample={do_sample}, temp={temperature}")
    
    # Step 2: Load data
    print(f"  Loading dataset...")
    dataset = ALEXSISDataset(data_path, generate_dummy_if_missing=False)
    
    # Split data if needed
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    test_indices = list(range(total_size - test_size, total_size))
    
    sources = []
    references = []
    
    for idx in test_indices:
        example = dataset[idx]
        sources.append(example["source"])
        references.append([example["reference"]])  # Wrap in list for SARI
    
    print(f"  ✓ Dataset loaded: {total_size} total examples")
    print(f"  ✓ Test set: {len(sources)} examples")
    if len(sources) > 0:
        print(f"  Sample source: {sources[0][:80]}...")
    
    # Step 3: Load model and tokenizer
    print(f"\n[STEP 3/6] Loading Model and Tokenizer")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
    
    # Get torch dtype from config
    torch_dtype_str = model_config.get("torch_dtype", "float32")
    if torch_dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype_str == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"  Precision: {torch_dtype_str}")
    print(f"  Loading model from: {model_path}")
    
    # Load model with precision specification
    use_device_map = device == "cuda" and torch.cuda.is_available()
    
    try:
        if use_device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )
    except Exception:
        # Fallback to AutoModel if not a causal LM
        if use_device_map:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )
    
    # Explicitly freeze all parameters (Frozen Backbone assumption)
    print(f"  Freezing model parameters...")
    num_params = sum(p.numel() for p in model.parameters())
    frozen_params = 0
    for param in model.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    
    # Move to device if not using device_map
    if not use_device_map:
        model = model.to(device)
    
    model.eval()
    print(f"  ✓ Model loaded: {num_params / 1e6:.2f}M parameters (all frozen)")
    
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ✓ Set pad_token to eos_token")
    
    print(f"  ✓ Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Step 4: Build profile from vocabulary
    print(f"\n[STEP 4/6] Building Difficulty Profile")
    print(f"  Extracting vocabulary from sources...")
    
    # Extract vocabulary from sources
    all_words = set()
    for source in sources:
        words = source.lower().split()
        words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        all_words.update(words)
    
    print(f"  ✓ Extracted {len(all_words)} unique words from vocabulary")
    
    # Build profile
    print(f"  Building profile with:")
    print(f"    - Frequency loader: {'file' if frequency_file else 'mock data'}")
    print(f"    - Long word threshold: {long_word_threshold} characters")
    print(f"    - Difficulty threshold: {threshold}")
    
    builder = ProfileBuilder(
        beta_Freq=beta_Freq,
        beta_Morph=beta_Morph,
        beta_Mask=beta_Mask,
        threshold=threshold,
        frequency_file=frequency_file,
        long_word_threshold=long_word_threshold,
    )
    
    profile = builder.build_profile(all_words)
    print(f"  ✓ Profile built:")
    print(f"    - Total words profiled: {len(profile.word_profiles)}")
    print(f"    - Difficult words (V_difficult): {len(profile.V_difficult)}")
    if len(profile.V_difficult) > 0:
        sample_difficult = list(profile.V_difficult)[:5]
        print(f"    - Sample difficult words: {', '.join(sample_difficult)}")
    
    # Build embeddings
    print(f"\n[STEP 5/6] Building Difficulty Embeddings")
    print(f"  Computing embeddings for {len(profile.V_difficult)} difficult words...")
    print(f"  Method: Average subword embeddings → L2 normalize")
    
    embedding_set = build_difficulty_embeddings(profile, tokenizer, model, device=device)
    print(f"  ✓ Embeddings built:")
    print(f"    - Embedding tensor shape: {embedding_set.E_diff.shape}")
    print(f"    - Hidden dimension: {embedding_set.hidden_dim}")
    print(f"    - Words with embeddings: {embedding_set.num_words}")
    
    # Step 5: Run generation
    print(f"\n[STEP 6/6] Generating Simplifications with DPGD")
    print(f"  Applying penalties during generation:")
    print(f"    - Semantic Distance Penalty (SDP): α={alpha_SDP}, δ={delta_SDP}")
    print(f"    - Frequency Penalty: α={alpha_Freq}")
    print(f"    - Morphology Penalty: α={alpha_Morph}")
    print(f"  Generation settings: max_tokens={max_new_tokens}, beams={num_beams}")
    
    predictions = []
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for i, source in enumerate(sources):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i + 1}/{len(sources)} ({100*(i+1)/len(sources):.1f}%) - {rate:.2f} ex/s")
            
            try:
                prompt = (
                    "Simplify this Spanish text for a user with limited reading experience.\n\n"
                    f"Text: {source}\n\n"
                    "Simplified text:"
                )
                pred = generate_with_dpgd(
                    model=model,
                    tokenizer=tokenizer,
                    profile=profile,
                    embedding_set=embedding_set,
                    input_text=prompt,
                    alpha_SDP=alpha_SDP,
                    alpha_Freq=alpha_Freq,
                    alpha_Morph=alpha_Morph,
                    delta_SDP=delta_SDP,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                predictions.append(pred)
            except Exception as e:
                print(f"  ✗ Error generating for example {i}: {e}")
                predictions.append("")  # Empty prediction on error
    
    elapsed = time.time() - start_time
    print(f"  ✓ Generation complete: {len(predictions)} predictions in {elapsed:.2f}s ({len(predictions)/elapsed:.2f} ex/s)")
    if len(predictions) > 0 and predictions[0]:
        print(f"  Sample prediction: {predictions[0][:80]}...")
    
    # Step 6: Compute metrics
    print(f"\n[METRICS] Computing Evaluation Metrics")
    
    # SARI
    print(f"  [1/4] Computing SARI (System output Against References and Input)...")
    sari_score = compute_sari(sources, predictions, references)
    print(f"    ✓ SARI: {sari_score:.4f}")
    
    # BERTScore
    print(f"  [2/4] Computing BERTScore (semantic similarity)...")
    # Flatten references for BERTScore (use first reference)
    refs_flat = [ref[0] if ref else "" for ref in references]
    bertscore_results = compute_bertscore(predictions, refs_flat, lang="es", device=device)
    print(f"    ✓ BERTScore - Precision: {bertscore_results['precision_mean']:.4f}, "
          f"Recall: {bertscore_results['recall_mean']:.4f}, "
          f"F1: {bertscore_results['f1_mean']:.4f}")
    
    # Readability
    print(f"  [3/4] Computing Readability (Szigriszt-Pazos index)...")
    source_readability = [compute_szigriszt_pazos(s) for s in sources]
    pred_readability = [compute_szigriszt_pazos(p) for p in predictions]
    readability_improvement = (sum(pred_readability) - sum(source_readability)) / len(source_readability)
    print(f"    ✓ Source readability: {sum(source_readability)/len(source_readability):.2f}")
    print(f"    ✓ Prediction readability: {sum(pred_readability)/len(pred_readability):.2f}")
    print(f"    ✓ Improvement: {readability_improvement:+.2f}")
    
    # PHR
    print(f"  [4/4] Computing Profile Hit Rate (PHR)...")
    phr_results = compute_phr_details(sources, predictions, profile)
    phr_score = compute_phr(sources, predictions, profile)
    print(f"    ✓ PHR: {phr_score:.4f}")
    print(f"    ✓ Violated words: {phr_results['total_violated']} / {phr_results['total_active']} active")
    if phr_results['violated_words']:
        print(f"    ✓ Violated words list: {', '.join(phr_results['violated_words'][:10])}")
    
    # Compile results
    results = {
        "config": {
            "model": model_name,
            "beta_Freq": beta_Freq,
            "beta_Morph": beta_Morph,
            "beta_Mask": beta_Mask,
            "threshold": threshold,
            "alpha_SDP": alpha_SDP,
            "alpha_Freq": alpha_Freq,
            "alpha_Morph": alpha_Morph,
        },
        "metrics": {
            "sari": float(sari_score),
            "bertscore": {
                "precision_mean": bertscore_results["precision_mean"],
                "recall_mean": bertscore_results["recall_mean"],
                "f1_mean": bertscore_results["f1_mean"],
            },
            "readability": {
                "source_mean": float(sum(source_readability) / len(source_readability)),
                "prediction_mean": float(sum(pred_readability) / len(pred_readability)),
                "improvement": float(
                    (sum(pred_readability) - sum(source_readability)) / len(source_readability)
                ),
            },
            "phr": float(phr_score),
            "phr_details": phr_results,
        },
        "profile_stats": {
            "total_words": len(profile.word_profiles),
            "difficult_words": len(profile.V_difficult),
        },
        "num_examples": len(sources),
    }
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[RESULTS] Saving Results")
    print(f"  ✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n📊 Metrics Summary:")
    print(f"  SARI:                    {sari_score:.4f} (higher is better)")
    print(f"  BERTScore F1:             {bertscore_results['f1_mean']:.4f} (higher is better)")
    print(f"  Readability Improvement:  {results['metrics']['readability']['improvement']:+.2f} (positive = easier)")
    print(f"  Profile Hit Rate (PHR):   {phr_score:.4f} (higher = fewer difficult words)")
    
    print(f"\n📈 Profile Statistics:")
    print(f"  Total words profiled:     {len(profile.word_profiles)}")
    print(f"  Difficult words:           {len(profile.V_difficult)}")
    print(f"  Examples evaluated:       {len(sources)}")
    
    print(f"\n⚙️  Configuration:")
    print(f"  Model:                     {model_name}")
    print(f"  β_Freq:                    {beta_Freq}")
    print(f"  β_Morph:                   {beta_Morph}")
    print(f"  β_Mask:                    {beta_Mask}")
    print(f"  α_SDP:                     {alpha_SDP}")
    print(f"  α_Freq:                    {alpha_Freq}")
    print(f"  α_Morph:                   {alpha_Morph}")
    print(f"  δ_SDP:                     {delta_SDP}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80 + "\n")
    
    return results

