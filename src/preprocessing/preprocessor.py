"""
Video Frame Preprocessing and Enhancement Module for Football Analytics.
Provides CLAHE contrast equalization, gamma correction, Gaussian/Bilateral denoising,
and Laplacian motion blur detection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """
    Container for preprocessed frame output and image quality diagnostics.

    Attributes:
        frame: Processed/enhanced BGR frame.
        blur_score: Laplacian variance blur metric (higher = sharper).
        is_blurry: Boolean flag indicating excessive motion blur.
        avg_brightness: Mean pixel luminance [0..255].
    """
    frame: np.ndarray
    blur_score: float
    is_blurry: bool
    avg_brightness: float


class FramePreprocessor:
    """
    Applies adaptive contrast enhancement, illumination correction, and noise reduction
    to football match broadcast video frames before object detection.
    """

    def __init__(
        self,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        enable_gamma: bool = False,
        gamma: float = 1.15,
        denoise_method: str = "none",  # "none", "gaussian", "bilateral"
        blur_threshold: float = 100.0,
        target_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size
        )
        self.enable_gamma = enable_gamma
        self.gamma = gamma
        self.denoise_method = denoise_method
        self.blur_threshold = blur_threshold
        self.target_resolution = target_resolution

        # Precompute gamma correction lookup table
        if self.enable_gamma and self.gamma > 0:
            inv_gamma = 1.0 / self.gamma
            self.gamma_table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype("uint8")
        else:
            self.gamma_table = None

    def calculate_blur(self, gray_frame: np.ndarray) -> float:
        """Calculate image sharpness using fast downscaled Laplacian operator."""
        if gray_frame.shape[0] > 360:
            small_gray = cv2.resize(gray_frame, (480, 270), interpolation=cv2.INTER_NEAREST)
        else:
            small_gray = gray_frame
        return float(cv2.Laplacian(small_gray, cv2.CV_16S).var())

    def process(self, frame: np.ndarray) -> PreprocessingResult:
        """
        Enhance a single video frame with contrast equalization and quality estimation.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        out = frame.copy()

        # 1. Optional target resizing
        if self.target_resolution is not None:
            out = cv2.resize(out, self.target_resolution, interpolation=cv2.INTER_LINEAR)

        # 2. High-speed luminance CLAHE in YCrCb color space (avoids expensive 3-channel split/merge)
        if self.enable_clahe:
            ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
            gray = ycrcb[:, :, 0]
            out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        # 3. Gamma correction (if enabled for underexposed stadiums)
        if self.enable_gamma and self.gamma_table is not None:
            out = cv2.LUT(out, self.gamma_table)

        # 4. Optional denoising
        if self.denoise_method == "gaussian":
            out = cv2.GaussianBlur(out, (3, 3), 0)
        elif self.denoise_method == "bilateral":
            out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)

        # 5. Image quality & blur metric (computed on luminance)
        blur_score = self.calculate_blur(gray)
        is_blurry = blur_score < self.blur_threshold
        avg_brightness = float(np.mean(gray))

        return PreprocessingResult(
            frame=out,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            avg_brightness=round(avg_brightness, 2),
        )
