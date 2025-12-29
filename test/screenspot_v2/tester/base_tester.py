from abc import ABC, abstractmethod
from typing import Optional, Tuple

from PIL import Image


class BaseTester(ABC):
    """Base interface for Screenspot testers.

    Implement `generate_click_coordinate` to return normalized (x, y) in [0, 1].
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def generate_click_coordinate(self, instruction: str, image: Image.Image) -> Optional[Tuple[float, float]]:
        """Return normalized click point or None when parsing fails."""

    @abstractmethod
    def _parse_output(self, response: str) -> Optional[Tuple[float, float]]:
        """Parse model/tool output and return normalized coordinates."""
