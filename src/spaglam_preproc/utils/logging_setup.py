# spaglam_preproc/utils/logging_setup.py

import logging
from rich.logging import RichHandler

def setup_logging(log_path: str):
    """
    Configures a rich logger to print to the console and a file.

    Args:
        log_path: Path to the output log file.
    """
    # Configure the rich handler for beautiful console output
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        tracebacks_suppress=[__import__("typer")], # Suppress typer's internal traceback frames
    )
    
    # Configure the file handler for persistent logging
    file_handler = logging.FileHandler(log_path, mode='w') # Overwrite log on each run
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s - %(message)s")
    )

    # Get the root logger and add handlers
    # We set the level on the handlers individually to control verbosity
    rich_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler, file_handler]
    )
