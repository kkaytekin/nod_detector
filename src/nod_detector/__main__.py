"""Main entry point for the nod-detector package.

This module provides a command-line interface for the nod-detector package.
It's called when the package is run using `python -m nod_detector`.
"""

import sys
from typing import List, Optional

from .cli import app


def main(args: Optional[List[str]] = None) -> None:
    """Run the nod-detector CLI application.

    Args:
        args: Command line arguments. If None, uses sys.argv.
    """
    if args is None:
        args = sys.argv[1:]
    app(args=args)


if __name__ == "__main__":
    main()
