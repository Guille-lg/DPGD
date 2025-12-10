# DPGD: Dynamic Profile-Guided Decoding for Lexical Simplification

A research implementation of **Dynamic Profile-Guided Decoding (DPGD)** for lexical simplification tasks, specifically designed for Spanish text simplification using large language models.

## What is DPGD?

DPGD is a decoding strategy that dynamically guides text generation to avoid complex words and prefer simpler alternatives. Unlike traditional fine-tuning approaches, DPGD operates at **inference time** by modifying the model's logits (token probabilities) during generation based on a user-specific difficulty profile.

### Key Features

- **No Fine-tuning Required**: Works with frozen pre-trained models
- **User-Adaptive**: Builds difficulty profiles from vocabulary
- **Real-time Decoding**: Applies penalties during generation
- **Multi-component Scoring**: Combines frequency, morphology, and semantic similarity
- **Spanish-Optimized**: Includes Spanish-specific morphological analysis

## How It Works

### Overview

DPGD modifies the generation process by:

1. **Building a Hybrid Difficulty Profile** from the input vocabulary
2. **Creating Difficulty Embeddings** for complex words
3. **Applying Penalties** during decoding to discourage difficult words
4. **Generating Simplified Text** that avoids complex vocabulary

### Mathematical Formulation

#### 1. Hybrid Difficulty Score ($S_w$)

For each surface word type $w$, DPGD computes a composite difficulty score:

$$
S_w = \beta_{\text{Freq}} \cdot F_{w,\text{norm}} +
      \beta_{\text{Morph}} \cdot M_w +
      \beta_{\text{Mask}} \cdot K_w
$$

Where:

- **$F_{w,\text{norm}} \in [0, 1]$** – normalized **frequency-based difficulty** score.  
  Higher values correspond to **rarer / more difficult** words (not to higher raw frequency).
- **$M_w \in [0, 1]$** – morphological complexity score (binary or scaled), capturing features like long words, complex derivational morphology, irregular forms, etc.
- **$K_w \in \{0, 1\}$** – **unknown-to-the-user flag**.  
  - $K_w = 1$: the word is explicitly considered *unknown* (not in frequency data or known word list) and should be hard-masked.  
  - $K_w = 0$: the word is *known* (appears in frequency data or provided known word list).
- **$\beta_{\text{Freq}}, \beta_{\text{Morph}}, \beta_{\text{Mask}} \ge 0$** – weights controlling the contribution of each component to $S_w$.

Words with $S_w \ge \theta$ (for a profile-dependent threshold $\theta$) form the difficult-vocabulary set
$V_{\text{difficult}} \subseteq W$.

#### 2. Hybrid Difficulty Embedding Set ($E_{\text{diff}}$)

For each difficult word type $w \in V_{\text{difficult}}$:

1. **Tokenize** with the same tokenizer used by the LLM:
   $$
   w \;\rightarrow\; \text{Tok}(w) = [t_1, t_2, \ldots, t_{L(w)}]
   $$
2. **Retrieve token embeddings** $e_{t_1}, \ldots, e_{t_{L(w)}}$ from the model’s static embedding matrix.
3. **Average to get a word embedding**:
   $$
   e_w = \frac{1}{L(w)} \sum_{j=1}^{L(w)} e_{t_j}
   $$
4. **Normalize** to unit length:
   $$
   \tilde e_w = \frac{e_w}{\|e_w\|_2}
   $$

Following the paper, the **Hybrid Difficulty Embedding Set** stores **both** the difficulty score and the normalized embedding:

$$
E_{\text{diff}} = \left\{ \left(S_w, \tilde e_w\right) \;|\; w \in V_{\text{difficult}} \right\}
$$

So $E_{\text{diff}}$ is a set of difficulty-weighted semantic prototypes, not just a plain embedding matrix.

#### 3. Semantic Distance Penalty (SDP)

During decoding, for each candidate token $t_i$ in the vocabulary, let $e_i$ be its embedding and
$\tilde e_i = e_i / \|e_i\|_2$ its $\ell_2$-normalized version.

We compare $\tilde e_i$ to each prototype $\tilde e_w$ in $E_{\text{diff}}$:

1. **Cosine similarity** to each difficult prototype:
   $$
   s_i(w) = \cos(\tilde e_i, \tilde e_w)
   $$
2. **Maximum similarity** over all difficult words:
   $$
   s_i^{\max} = \max_{w \in V_{\text{difficult}}} s_i(w)
   $$
   and the corresponding nearest difficult word
   $$
   w^\star(e_i) = \arg\max_{w \in V_{\text{difficult}}} s_i(w)
   $$
3. **Cosine distance** to the nearest difficult prototype:
   $$
   D_{\min}(e_i) = 1 - s_i^{\max}
   $$
   (since cosine similarity lies in $[-1,1]$, $D_{\min}(e_i) \in [0,2]$).

Given a distance margin $\delta \in (0, 2]$ and a scaling factor $\gamma \ge 0$, the **Semantic Distance Penalty**
for token $t_i$ is:

$$
P_{\text{SDP}}(t_i)
= \gamma \, S_{w^\star(e_i)} \, \max\big(0,\, \delta - D_{\min}(e_i)\big)
$$

- Tokens whose embeddings lie **within** the margin of some difficult prototype
  ($D_{\min}(e_i) < \delta$) receive a positive penalty that grows as:
  - they get closer in embedding space to that prototype, and
  - the prototype’s scalar difficulty $S_{w^\star(e_i)}$ increases.
- Tokens with $D_{\min}(e_i) \ge \delta$ have $P_{\text{SDP}}(t_i) = 0$ and are not penalized by SDP.

By convention, when $V_{\text{difficult}} = \emptyset$, we define $P_{\text{SDP}}(t_i) = 0$ for all tokens, so that
DPGD reduces exactly to the baseline LLM decoding.

#### 4. Profile-Linked Logit Adjustment

At each decoding step $k$, the LLM produces a vector of logits $z^{(k)} \in \mathbb{R}^{|V|}$, where $z^{(k)}_i$
is the logit for token $t_i$ given the current prefix.

DPGD computes a **per-token penalty** $P(t_i)$ combining SDP with deterministic profile scores:

- Let $w_i^{(k)}$ be the **context-dependent difficult word** completed by $t_i$ at step $k$, according
  to the token–word alignment logic from the paper (only certain prefixes + $t_i$ correspond to a
  difficult surface word type). If no difficult word is aligned, set $w_i^{(k)} = \emptyset$.

Then for tokens that **complete a difficult word** ($w_i^{(k)} \in V_{\text{difficult}}$):

$$
P(t_i) = \alpha_{\text{SDP}} \cdot P_{\text{SDP}}(t_i)
       + \alpha_{\text{Freq}} \cdot F_{w_i,\text{norm}}
       + \alpha_{\text{Morph}} \cdot M_{w_i}
       + C_{\text{mask}} \cdot \mathbf{1}[K_{w_i} = 1]
$$

Where:

- $\alpha_{\text{SDP}}, \alpha_{\text{Freq}}, \alpha_{\text{Morph}} \ge 0$ are weights for semantic, frequency, and morphology components.
- $C_{\text{mask}} \gg 0$ is a large constant used for **hard masking**:  
  if $K_{w_i} = 1$ (word unknown to the user), the penalty is large enough that the logit
  becomes effectively $-\infty$ after scaling.
- $\mathbf{1}[\cdot]$ is the indicator function.

For tokens **not aligned** to any difficult word ($w_i^{(k)} = \emptyset$), we treat:

$$
F_{w_i,\text{norm}} = 0,\quad M_{w_i} = 0,\quad K_{w_i} = 0
$$

so that the penalty reduces to the **pure semantic SDP term**:

$$
P(t_i) = \alpha_{\text{SDP}} \cdot P_{\text{SDP}}(t_i)
$$

Finally, the **modified logit** for token $t_i$ at step $k$ is:

$$
z_i^{\prime (k)} = z_i^{(k)} - \lambda \cdot P(t_i)
$$

with a global personalization strength $\lambda \ge 0$. The generation step then samples from the
temperature-scaled softmax of the adjusted logits:

$$
p_i^{(k)} = \text{softmax}\Big(\frac{z^{\prime (k)}}{\tau}\Big)_i
$$

where $\tau > 0$ is the decoding temperature.

#### 5. Profile Hit Rate (PHR)

PHR measures how well the system **avoids difficult words** specified in the profile.

Let:

- $\text{WordSet}(X)$: set of word types appearing in the source sentence.
- $\text{WordSet}(Y)$: set of word types appearing in the generated output.
- $V_{\text{difficult}}$: difficult word types in the profile.

Then:

- **Active difficult words** for an example:
  $$
  V_{\text{active}}(X, Y, P_u)
  = V_{\text{difficult}} \cap \big(\text{WordSet}(X) \cup \text{WordSet}(Y)\big)
  $$
- **Violated difficult words** (those that appear in the output):
  $$
  V_{\text{violated}}(Y, P_u)
  = V_{\text{difficult}} \cap \text{WordSet}(Y)
  $$

For examples where $|V_{\text{active}}(X, Y, P_u)| > 0$, the **sentence-level** PHR is:

$$
\text{PHR}(X, Y, P_u)
= 1 - \frac{\lvert V_{\text{violated}}(Y, P_u) \rvert}
         {\lvert V_{\text{active}}(X, Y, P_u) \rvert}
$$

Corpus-level PHR is obtained by averaging this score over all examples where
$|V_{\text{active}}(X, Y, P_u)| > 0$, i.e., where the profile is actually relevant.


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd LexicalSimplification

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/feralvam/easse.git
```

### Dependencies

- `torch>=2.0.0`: PyTorch for model operations
- `transformers>=4.30.0`: HuggingFace transformers
- `bert-score>=0.3.13`: BERTScore metric
- `pyyaml>=6.0`: Configuration file parsing
- `typer>=0.9.0`: CLI framework
- `pydantic>=2.0.0`: Data validation

**Note:** `easse` is optional - the code includes a fallback SARI implementation.

## Project Structure

```
LexicalSimplification/
├── configs/
│   ├── model/              # Model configurations (e.g., Llama-7B-ES)
│   ├── profiles/           # Profile configurations
│   └── experiments/        # Experiment configurations
├── src/
│   └── dpgd/
│       ├── __init__.py     # Package initialization
│       ├── cli.py          # Command-line interface
│       ├── config.py       # Configuration loader
│       ├── data/           # Data handling
│       │   ├── alexsis_dataset.py    # ALEXSIS dataset loader
│       │   ├── preprocessing.py      # Text standardization
│       │   └── tokenization.py       # Tokenization utilities
│       ├── decoding/       # Decoding components
│       │   ├── dpgd_logits_processor.py  # Core DPGD logits processor
│       │   └── generation.py            # Generation wrapper
│       ├── embeddings/     # Embedding utilities
│       │   ├── embedding_cache.py       # Embedding matrix extraction
│       │   └── difficulty_embeddings.py # Difficulty embedding construction
│       ├── experiments/    # Experiment runners
│       │   └── run_eval.py              # Full evaluation pipeline
│       ├── metrics/        # Evaluation metrics
│       │   ├── sari.py                  # SARI metric
│       │   ├── bertscore.py             # BERTScore metric
│       │   ├── readability_es.py        # Spanish readability (Szigriszt-Pazos)
│       │   └── profile_hit_rate.py      # PHR metric
│       ├── profiles/       # Profile building
│       │   ├── profile_schema.py        # Profile data structures
│       │   ├── profile_builder.py       # Profile factory
│       │   ├── frequency_loader.py      # Frequency data loader
│       │   └── morphology_analyzer.py   # Morphological complexity analysis
│       └── utils/
│           └── io_utils.py  # I/O utilities (YAML, JSON)
├── requirements.txt
└── README.md
```

## Code Documentation

### Core Modules

#### `src/dpgd/cli.py`
Command-line interface using Typer. Provides commands:
- `eval`: Run full evaluation pipeline
- `preprocess`: Preprocess datasets
- `version`: Show version information

#### `src/dpgd/config.py`
Configuration management system. `ConfigLoader` class:
- Loads and merges multiple YAML configuration files
- Supports deep merging of nested dictionaries
- Provides dot-notation access to config values

#### `src/dpgd/data/alexsis_dataset.py`
ALEXSIS dataset loader. `ALEXSISDataset` class:
- Loads JSONL format datasets with "source" and "reference" fields
- Generates dummy data if file is missing (for testing)
- Provides iterator interface and helper methods

#### `src/dpgd/data/preprocessing.py`
Text preprocessing utilities:
- `standardize_text()`: Normalizes quotes, whitespace, Unicode
- `preprocess_dataset()`: Batch preprocessing for JSONL files

#### `src/dpgd/data/tokenization.py`
Tokenization utilities:
- `TokenizerWrapper`: Wrapper around HuggingFace tokenizers
- `get_word_to_token_alignment()`: Maps surface words to subword token indices
- Critical for applying penalties to specific word parts

#### `src/dpgd/profiles/profile_schema.py`
Profile data structures:
- `WordProfile`: Stores F_norm, M_w, K_w, S_w for a word
- `HybridDifficultyProfile`: Container for all word profiles and V_difficult set

#### `src/dpgd/profiles/profile_builder.py`
Profile factory class:
- `ProfileBuilder`: Builds profiles from vocabulary
- Integrates frequency loader and morphology analyzer
- Computes composite scores and identifies difficult words

#### `src/dpgd/profiles/frequency_loader.py`
Frequency data management:
- `FrequencyLoader`: Loads word frequencies from CSV
- Generates mock frequencies if file missing
- Provides normalized frequency scores (F_norm)

#### `src/dpgd/profiles/morphology_analyzer.py`
Morphological complexity analysis:
- `MorphologyAnalyzer`: Analyzes Spanish word complexity
- Detects long words, complex suffixes/prefixes
- Returns morphological complexity score (M_w)

#### `src/dpgd/embeddings/embedding_cache.py`
Embedding extraction utilities:
- `extract_embedding_matrix()`: Extracts model's input embedding matrix
- `normalize_embeddings()`: L2 normalization
- Helper functions for token embedding retrieval

#### `src/dpgd/embeddings/difficulty_embeddings.py`
Difficulty embedding construction:
- `DifficultyEmbeddingSet`: Container for difficulty embeddings and scores
- `build_difficulty_embeddings()`: Constructs E_diff from profile
- Averages subword embeddings and normalizes

#### `src/dpgd/decoding/dpgd_logits_processor.py`
**Core DPGD implementation**. `DPGDLogitsProcessor` class:
- Inherits from `transformers.LogitsProcessor`
- `__call__()`: Applies penalties to logits during generation
- Implements SDP, frequency, and morphology penalties
- Optimized word boundary detection (only decodes last N tokens)

#### `src/dpgd/decoding/generation.py`
Generation wrapper:
- `generate_with_dpgd()`: Wraps `model.generate()` with DPGD processor
- Handles batching, device management, tokenization
- Returns generated text(s)

#### `src/dpgd/experiments/run_eval.py`
Full evaluation pipeline:
- `run_evaluation()`: Complete experiment runner
- Loads config, data, model
- Builds profiles and embeddings
- Runs generation and computes all metrics
- Saves results to JSON

#### `src/dpgd/metrics/sari.py`
SARI (System output Against References and Input) metric:
- Measures word-level operations (keep, add, delete)
- Uses `easse` library if available, falls back to simple implementation

#### `src/dpgd/metrics/bertscore.py`
BERTScore metric wrapper:
- Computes semantic similarity using BERT embeddings
- Returns precision, recall, F1 scores

#### `src/dpgd/metrics/readability_es.py`
Spanish readability metric:
- `compute_szigriszt_pazos()`: Implements Szigriszt-Pazos index
- Formula: $SP = 206.84 - (62.3 \cdot \frac{\text{syllables}}{\text{words}}) - \frac{\text{words}}{\text{sentences}}$
- Higher scores = easier text

#### `src/dpgd/metrics/profile_hit_rate.py`
Profile Hit Rate (PHR) metric:
- `compute_phr()`: Computes PHR score
- `compute_phr_details()`: Returns detailed statistics
- Measures how well system avoids difficult words

## Usage

### Quick Start

```bash
# Run evaluation with default config
python -m src.dpgd.cli eval configs/experiments/placeholder.yaml

# With custom output directory
python -m src.dpgd.cli eval configs/experiments/placeholder.yaml -o ./my_results

# With explicit device
python -m src.dpgd.cli eval configs/experiments/placeholder.yaml --device cuda
```

### Configuration

Create an experiment configuration file (YAML):

```yaml
experiment:
  name: "my_experiment"
  max_new_tokens: 100
  num_beams: 1
  do_sample: true
  temperature: 0.7

model:
  name: "Llama-7B-ES"
  path: "Kukedlc/Llama-7b-spanish"
  torch_dtype: "bfloat16"

profile:
  beta_Freq: 0.4
  beta_Morph: 0.4
  beta_Mask: 0.2
  threshold: 0.5
  alpha_SDP: 1.0
  alpha_Freq: 1.0
  alpha_Morph: 1.0
  delta_SDP: 0.0

data:
  path: "data/alexsis.jsonl"
  test_split: 1.0
```

### Programmatic Usage

```python
from src.dpgd.profiles import ProfileBuilder
from src.dpgd.embeddings import build_difficulty_embeddings
from src.dpgd.decoding import generate_with_dpgd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("dccuchile/llama-7b-es")
tokenizer = AutoTokenizer.from_pretrained("dccuchile/llama-7b-es")

# Build profile
builder = ProfileBuilder(beta_Freq=0.4, beta_Morph=0.4, beta_Mask=0.2, threshold=0.5)
vocab = ["inconstitucionalmente", "desinstitucionalización", "el", "la"]
profile = builder.build_profile(vocab)

# Build embeddings
embedding_set = build_difficulty_embeddings(profile, tokenizer, model)

# Generate
source = "El tribunal declaró inconstitucionalmente la ley."
simplified = generate_with_dpgd(
    model=model,
    tokenizer=tokenizer,
    profile=profile,
    embedding_set=embedding_set,
    input_text=source,
    alpha_SDP=1.0,
    alpha_Freq=1.0,
    alpha_Morph=1.0,
)
```

## Evaluation Metrics

The system computes four metrics:

1. **SARI**: Measures word-level operations (keep/add/delete)
2. **BERTScore**: Semantic similarity using BERT embeddings
3. **Readability (Szigriszt-Pazos)**: Spanish readability index
4. **PHR (Profile Hit Rate)**: Measures avoidance of difficult words

Results are saved to `results/results.json` with detailed statistics.

## Performance Optimizations

- **Partial Decoding**: Only decodes last 20 tokens for word boundary detection (50x+ speedup)
- **Cache Management**: Limits decoded cache to 1000 entries
- **Memory Efficiency**: Uses bfloat16 for 7B models (50% memory reduction)
- **Frozen Backbone**: Model parameters explicitly frozen (no gradient computation)

## Citation

If you use this code, please cite the original DPGD paper:

```bibtex
@article{dpgd2024,
  title={Dynamic Profile-Guided Decoding for Lexical Simplification},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Specify your license here]

## Contributing

This is a research prototype. Contributions and improvements are welcome!

## Acknowledgments

- Based on the DPGD research paper
- Uses HuggingFace transformers
- Spanish morphological analysis inspired by linguistic research
