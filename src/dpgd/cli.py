"""Command-line interface for DPGD."""

from pathlib import Path
from typing import Optional

import typer

from .config import ConfigLoader
from .data.preprocessing import preprocess_dataset
from .experiments.run_eval import run_evaluation

app = typer.Typer(
    name="dpgd",
    help="Dynamic Profile-Guided Decoding for Lexical Simplification",
    add_completion=False,
)


@app.command("run-eval")
def run_eval(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the main configuration file",
        exists=True,
    ),
    model_config: Optional[Path] = typer.Option(
        None,
        "--model-config",
        help="Path to the model configuration file",
        exists=True,
    ),
    profile_config: Optional[Path] = typer.Option(
        None,
        "--profile-config",
        help="Path to the profile configuration file",
        exists=True,
    ),
    experiment_config: Optional[Path] = typer.Option(
        None,
        "--experiment-config",
        help="Path to the experiment configuration file",
        exists=True,
    ),
) -> None:
    """
    Run evaluation with the specified configuration files.
    
    If --config is provided, it will be loaded. Additional config files
    (model, profile, experiment) can be provided to override or extend
    the base configuration.
    """
    loader = ConfigLoader()
    
    config_paths = []
    if config:
        config_paths.append(config)
    if model_config:
        config_paths.append(model_config)
    if profile_config:
        config_paths.append(profile_config)
    if experiment_config:
        config_paths.append(experiment_config)
    
    if not config_paths:
        typer.echo("Error: At least one configuration file must be provided.", err=True)
        raise typer.Exit(code=1)
    
    try:
        merged_config = loader.load(*config_paths)
        typer.echo("Config loaded successfully")
        typer.echo(f"Loaded {len(config_paths)} configuration file(s)")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("preprocess")
def preprocess(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input JSONL dataset file",
        exists=True,
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Path to output preprocessed JSONL file",
    ),
    source_field: str = typer.Option(
        "source",
        "--source-field",
        help="Name of the source text field in the dataset",
    ),
    reference_field: str = typer.Option(
        "reference",
        "--reference-field",
        help="Name of the reference text field in the dataset",
    ),
) -> None:
    """
    Preprocess a dataset by standardizing text (normalizing quotes, whitespace, etc.).
    
    This command reads a JSONL file, standardizes the text fields, and writes
    the preprocessed data to a new file.
    """
    try:
        preprocess_dataset(
            str(input_path),
            str(output_path),
            source_field=source_field,
            reference_field=reference_field,
        )
        typer.echo(f"Preprocessing completed successfully!")
        typer.echo(f"Input: {input_path}")
        typer.echo(f"Output: {output_path}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during preprocessing: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("eval")
def eval(
    config: Path = typer.Argument(
        ...,
        help="Path to experiment configuration file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        "./results",
        "--output-dir",
        "-o",
        help="Directory to save evaluation results",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to run on (e.g., 'cuda', 'cpu'). Auto-detects if not specified.",
    ),
) -> None:
    """
    Run full evaluation pipeline.
    
    This command:
    1. Loads config and dataset
    2. Builds difficulty profile and embeddings
    3. Runs generation with DPGD
    4. Computes SARI, BERTScore, Readability, and PHR metrics
    5. Saves results to JSON file
    """
    try:
        results = run_evaluation(
            config_path=config,
            output_dir=output_dir,
            device=device,
        )
        typer.echo("\n✅ Evaluation completed successfully!")
        typer.echo(f"Results saved to: {output_dir}/results.json")
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        import traceback
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1)


@app.command("version")
def version() -> None:
    """Print the version of DPGD."""
    from . import __version__
    typer.echo(f"DPGD version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

