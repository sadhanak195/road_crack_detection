"""Image input/output and URL handling module."""

import io
import re
from typing import Tuple, Optional, List
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps
import requests

from utils.logging_setup import get_logger

logger = get_logger("image_io")

MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
DEFAULT_TIMEOUT_SEC = 10
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class ImageProcessingError(Exception):
    """Custom exception for image processing and loading failures."""
    pass


def fix_orientation(image: Image.Image) -> Image.Image:
    """Correct EXIF orientation if present."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception as e:
        logger.debug(f"Failed to transpose EXIF orientation: {e}")
        return image


def pil_to_cv2(pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PIL image to RGB and BGR numpy arrays."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def cv2_to_pil(cv2_bgr: np.ndarray) -> Image.Image:
    """Convert BGR OpenCV image to PIL Image."""
    rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image_from_bytes(data: bytes) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """Load and validate image from raw bytes.

    Returns:
        (PIL.Image, rgb_ndarray, bgr_ndarray)
    """
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise ImageProcessingError(
            f"Image size ({len(data) / 1024 / 1024:.2f} MB) exceeds maximum allowed limit of 15 MB."
        )

    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img = fix_orientation(pil_img)
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise ImageProcessingError(f"Failed to decode image data: {str(e)}")

    rgb, bgr = pil_to_cv2(pil_img)
    return pil_img, rgb, bgr


def extract_html_candidate_images(html: str, base_url: str) -> List[str]:
    """Extract candidate image URLs from HTML page (og:image and <img> tags)."""
    candidates = []

    # Check for OpenGraph image
    og_matches = re.findall(
        r'<meta\s+property=["\']og:image["\']\s+content=["\'](.*?)["\']',
        html,
        re.IGNORECASE,
    )
    if not og_matches:
        og_matches = re.findall(
            r'<meta\s+content=["\'](.*?)["\']\s+property=["\']og:image["\']',
            html,
            re.IGNORECASE,
        )
    for m in og_matches:
        candidates.append(urljoin(base_url, m.strip()))

    # Check for <img> tags
    img_matches = re.findall(r'<img[^>]+src=["\'](.*?)["\']', html, re.IGNORECASE)
    for m in img_matches:
        url = urljoin(base_url, m.strip())
        if not url.startswith("data:") and url not in candidates:
            candidates.append(url)

    return candidates


def load_image_from_url(url: str) -> Tuple[Image.Image, np.ndarray, np.ndarray, str]:
    """Fetch and load image from URL.

    Handles direct image downloads, HTML pages with og:image/img fallback,
    and returns (PIL Image, rgb_np, bgr_np, resolved_url).
    """
    url = url.strip()
    if not url:
        raise ImageProcessingError("URL cannot be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageProcessingError(f"Unsupported URL scheme '{parsed.scheme}'. Please use http:// or https://")

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=DEFAULT_TIMEOUT_SEC,
            stream=True,
        )
    except requests.exceptions.Timeout:
        raise ImageProcessingError(f"Connection timed out ({DEFAULT_TIMEOUT_SEC}s) while fetching image from {url}")
    except requests.exceptions.SSLError as e:
        raise ImageProcessingError(f"SSL certificate verification failed for {url}: {e}")
    except requests.exceptions.RequestException as e:
        raise ImageProcessingError(f"Failed to fetch URL {url}: {str(e)}")

    if response.status_code != 200:
        raise ImageProcessingError(f"HTTP request failed with status code {response.status_code} ({response.reason})")

    content_type = response.headers.get("Content-Type", "").lower()
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_IMAGE_SIZE_BYTES:
        raise ImageProcessingError(
            f"Image size reported ({int(content_length) / 1024 / 1024:.2f} MB) exceeds 15 MB limit."
        )

    # If it is a direct image
    if content_type.startswith("image/"):
        raw_data = response.content
        pil_img, rgb, bgr = load_image_from_bytes(raw_data)
        return pil_img, rgb, bgr, url

    # If it returned HTML or other content, attempt fallback
    if "text/html" in content_type:
        html_text = response.text
        candidates = extract_html_candidate_images(html_text, url)
        if not candidates:
            raise ImageProcessingError(
                f"The provided URL is an HTML webpage ({content_type}) and no extractable images were found."
            )

        logger.info(f"HTML page returned. Found {len(candidates)} candidate images. Trying first candidate: {candidates[0]}")
        # Try fetching candidate images in order
        for candidate_url in candidates[:3]:
            try:
                cand_resp = requests.get(
                    candidate_url,
                    headers=HEADERS,
                    timeout=DEFAULT_TIMEOUT_SEC,
                )
                if cand_resp.status_code == 200 and cand_resp.headers.get("Content-Type", "").lower().startswith("image/"):
                    pil_img, rgb, bgr = load_image_from_bytes(cand_resp.content)
                    return pil_img, rgb, bgr, candidate_url
            except Exception as e:
                logger.warning(f"Failed fetching candidate {candidate_url}: {e}")
                continue

        raise ImageProcessingError(
            f"The URL returned an HTML page and could not retrieve valid images from candidates: {candidates[:3]}"
        )

    raise ImageProcessingError(
        f"Unsupported Content-Type '{content_type}'. Expected an image (e.g. image/jpeg, image/png)."
    )


def load_image_from_file_path(path: str) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """Load image from local disk path."""
    with open(path, "rb") as f:
        data = f.read()
    return load_image_from_bytes(data)
