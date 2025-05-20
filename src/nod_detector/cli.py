"""Command-line interface for the nod-detector package."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from nod_detector.pipeline.video_processing_pipeline import VideoProcessingPipeline

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=False,
            show_path=False,
        )
    ],
)
logger = logging.getLogger("nod_detector")

# Default callback for the app
app = typer.Typer(help="Nod Detector - Detect nodding behavior in videos")

# Create a console for rich output with safe encoding
console = Console(force_terminal=False, color_system=None)

# Default values for arguments
DEFAULT_INPUT_HELP = "Path to the input video file"
DEFAULT_OUTPUT_HELP = "Path to save the output video"
DEFAULT_VISUALIZE_HELP = "Enable visualization of the processing"
DEFAULT_DEBUG_HELP = "Run in debug mode (process only first 10 frames)"
DEFAULT_LOG_LEVEL_HELP = "Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"

# Default argument values to avoid B008 errors
DEFAULT_INPUT_ARG = typer.Argument(..., help=DEFAULT_INPUT_HELP)
DEFAULT_OUTPUT_OPTION = typer.Option(None, "--output", "-o", help=DEFAULT_OUTPUT_HELP)
DEFAULT_VISUALIZE_OPTION = typer.Option(False, "--visualize", "-v", help=DEFAULT_VISUALIZE_HELP)
DEFAULT_DEBUG_OPTION = typer.Option(False, "--debug", "-d", help=DEFAULT_DEBUG_HELP)
DEFAULT_LOG_LEVEL_OPTION = typer.Option("INFO", "--log-level", "-l", help=DEFAULT_LOG_LEVEL_HELP)


def validate_input_file(input_path: Path) -> Path:
    """Validate that the input file exists and is a video file."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input file '{input_path}' does not exist.")
    if not input_path.is_file():
        raise typer.BadParameter(f"'{input_path}' is not a file.")
    if input_path.suffix.lower() not in [".mp4", ".avi", ".mov"]:
        console.print(
            "[yellow]Warning:[/] Input file may not be a supported video format. " "Supported formats: .mp4, .avi, .mov",
            style="yellow",
        )
    return input_path


def ensure_output_path(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Ensure an output path is provided or generate a default one."""
    if output_path is None:
        return input_path.with_stem(f"{input_path.stem}_processed")
    return output_path


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with the specified log level.

    Args:
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL) or a typer.Option object
    """
    # Handle case where level is a typer.Option object
    if hasattr(level, "value"):
        level = level.value
    elif not isinstance(level, str):
        level = "INFO"

    log_level = getattr(logging, str(level).upper(), logging.INFO)

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create a timestamped log file
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nod_detector_{timestamp}.log"

    # Configure console handler with Rich
    console_handler = RichHandler(console=console, rich_tracebacks=True, markup=True, show_time=True, show_path=True, log_time_format="[%X]")
    console_handler.setLevel(log_level)

    # Configure file handler with more detailed format
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to ensure all messages are processed
        handlers=[console_handler, file_handler],
        force=True,  # Override any existing handlers
    )

    # Set log levels for specific loggers
    loggers = ["nod_detector", "mediapipe", "__main__", "mediapipe_components", "video_processing_pipeline", "cli"]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        # Ensure the logger propagates to the root logger
        logger.propagate = True

    # Log the log file location
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file.absolute()}")
    logger.info(f"Log level set to: {logging.getLevelName(log_level)}")


@app.command()
def process(
    input_file: Path = DEFAULT_INPUT_ARG,
    output: Optional[Path] = DEFAULT_OUTPUT_OPTION,
    visualize: bool = DEFAULT_VISUALIZE_OPTION,
    debug: bool = DEFAULT_DEBUG_OPTION,
    log_level: str = DEFAULT_LOG_LEVEL_OPTION,
) -> None:
    """Process a video file to detect nodding behavior.

    Args:
        input_file: Path to the input video file
        output: Path to save the output video (optional)
        visualize: Enable visualization of the processing
        debug: Run in debug mode (process only first 10 frames)
        log_level: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    setup_logging(level=log_level)
    logger.info(f"Starting nod detection on video: {input_file}")

    try:
        # Validate input file
        input_path = validate_input_file(input_file)

        # Set up output path
        output_path = ensure_output_path(input_path, output)

        # Initialize the pipeline
        pipeline = VideoProcessingPipeline()
        logger.info("Video processing pipeline initialized")

        # Process the video with visualization and debug flags
        result = pipeline.process(input_data=str(input_path), visualize=visualize, debug=debug, output_path=str(output_path))

        logger.info(f"Processing complete. Results saved to: {output_path}")
        console.print("\n[green]Processing complete![/]")
        console.print(f"\nOutput saved to: {output_path}")

        return result

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        sys.exit(1)


# The process function is the entry point for the CLI

if __name__ == "__main__":
    app()
