# spaglam_preproc/cli.py

import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console

from .core.dataset_writer import create_dataset_shards
from .utils.logging_setup import setup_logging

app = typer.Typer(
    name="spaglam-preproc",
    help="A high-performance, single-pass preprocessing pipeline for SpaGLaM.",
    add_completion=False,
)
console = Console()

@app.command()
def run(
    config_path: Path = typer.Option(
        ..., 
        "--config", 
        "-c",
        help="Path to the YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    )
):
    """
    Run the full SpaGLaM data preprocessing pipeline using a configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Error parsing config file '{config_path}':[/bold red] {e}")
        raise typer.Exit(code=1)
    
    # Ensure output directory exists before setting up logging
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file and console
    log_file_name = config.get('qc', {}).get('log_file_name', 'preprocessing.log')
    log_path = output_dir / log_file_name
    setup_logging(str(log_path))
    
    # Log the configuration for reproducibility
    logging.info("🚀 Starting SpaGLaM preprocessing pipeline with configuration:")
    # Pretty print the config to the log file
    logging.info("\n" + json.dumps(config, indent=2))
    
    try:
        create_dataset_shards(config)
        console.print(f"\n[bold green]✅ Pipeline finished successfully! Check outputs in '{output_dir}'.[/bold green]")
    except Exception:
        # The rich handler will automatically log the traceback
        logging.error("Pipeline failed with an unhandled exception.")
        console.print(f"\n[bold red]❌ Pipeline failed. See full traceback in the log file: {log_path}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
