"""Command-line interface for the nod-detector package."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nod_detector.pipeline.video_processing_pipeline import VideoProcessingPipeline

# Default callback for the app
app = typer.Typer(help="Nod Detector - Detect nodding behavior in videos")

# Create a console for rich output
console = Console()

# Default values for arguments
DEFAULT_INPUT_HELP = "Path to the input video file"
DEFAULT_OUTPUT_HELP = "Path to save the output video"
DEFAULT_VISUALIZE_HELP = "Enable visualization of the processing"

# Default argument values to avoid B008 errors
DEFAULT_INPUT_ARG = typer.Argument(..., help=DEFAULT_INPUT_HELP)
DEFAULT_OUTPUT_OPTION = typer.Option(None, "--output", "-o", help=DEFAULT_OUTPUT_HELP)
DEFAULT_VISUALIZE_OPTION = typer.Option(False, "--visualize", "-v", help=DEFAULT_VISUALIZE_HELP)


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


@app.command("process-video")
def process_video(
    input: Path = DEFAULT_INPUT_ARG,
    output: Optional[Path] = DEFAULT_OUTPUT_OPTION,
    visualize: bool = DEFAULT_VISUALIZE_OPTION,
) -> None:
    """Process a video file to detect nodding behavior."""
    try:
        # Validate input file
        input_path = validate_input_file(input)

        # Ensure output path
        output_path = ensure_output_path(input_path, output)

        # Initialize the pipeline
        pipeline = VideoProcessingPipeline()

        # Process the video
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Processing video...", total=None)

            # Process the video
            result = pipeline.process(str(input_path))

            # Update progress
            progress.update(task, completed=1, description="Processing complete!")

        # Show results
        console.print("\n✅ Processing complete!")
        console.print(f"\n📊 Results: {result}")
        console.print(f"\n💾 Output saved to: {output_path}")

    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
