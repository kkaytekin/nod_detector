"""Base pipeline module for the nod detector."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

logger = logging.getLogger(__name__)

# Define a type variable for the return type of _process
T = TypeVar("T", bound=Dict[str, Any])


class BasePipeline(ABC, Generic[T]):
    """Abstract base class for processing pipelines.

    This class defines the interface that all pipeline implementations should follow.

    Type Variables:
        T: The return type of the _process method, must be a dictionary.
    """

    def __init__(self) -> None:
        """Initialize the pipeline with default parameters."""
        self._is_running = False

    def process(self, input_data: Any, **kwargs: Any) -> T:
        """Process the input data through the pipeline.

        Args:
            input_data: The input data to process.
            **kwargs: Additional parameters specific to the pipeline implementation.

        Returns:
            The processing results as a dictionary.

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        # Check if the pipeline is already running
        if self._is_running:
            raise RuntimeError("Pipeline is already running")

        self._is_running = True
        try:
            return self._process(input_data, **kwargs)
        except Exception as e:
            logger.error("Error processing pipeline: %s", str(e), exc_info=True)
            raise
        finally:
            self._is_running = False

    @abstractmethod
    def _process(self, input_data: Any, **kwargs: Any) -> T:
        """Internal processing method to be implemented by subclasses.

        Args:
            input_data: The input data to process.
            **kwargs: Additional parameters specific to the pipeline implementation.

        Returns:
            The processing results as a dictionary.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self) -> None:
        """Reset the pipeline state."""
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Return whether the pipeline is currently running.

        Returns:
            bool: True if the pipeline is running, False otherwise.
        """
        return self._is_running
