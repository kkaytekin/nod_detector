#!/usr/bin/env python3
"""
Main entry point for the nod detection system.
"""
import argparse
import logging
import sys

from nod_detector.pipeline.video_processing_pipeline import VideoProcessingPipeline


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("nod_detector.log")],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Nod Detection System")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to save the output video (default: output.mp4)",
    )
    return parser.parse_args()


def main() -> int:
    """Main function to run the nod detection pipeline.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = parse_arguments()

    # Initialize the pipeline
    logger.info("Initializing video processing pipeline...")
    pipeline = VideoProcessingPipeline()

    try:
        # Process the video
        logger.info(f"Processing video: {args.input}")
        results = pipeline.process(args.input)

        # Log summary
        logger.info(f"Processing complete. Processed {len(results['frame_results'])} frames.")

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
