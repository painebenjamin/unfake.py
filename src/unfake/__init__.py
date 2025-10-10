"""
unfake - AI-generated pixel art optimization tool

A high-performance tool for improving AI-generated pixel art through scale detection,
color quantization, and smart downscaling. Features Rust acceleration for critical
operations.
"""

__version__ = "1.0.0"

# Import content-adaptive downscaling
from .content_adaptive import content_adaptive_downscale

# Import main functions from pixel module
from .pixel import (
    alpha_binarization,
    count_colors,
    detect_optimal_color_count,
    downscale_block,
    downscale_by_dominant_color,
    edge_aware_detect,
    extract_palette,
    finalize_pixels,
    jaggy_cleaner,
    main,
    morphological_cleanup,
    process_image,
    process_image_sync,
    quantize_colors,
    runs_based_detect,
)

# Import accelerated versions from rust integration
from .pixel_rust_integration import (
    RUST_AVAILABLE,
    WuQuantizerAccelerated,
    map_pixels_to_palette_accelerated,
    runs_based_detect_accelerated,
)

__all__ = [
    # Main processing functions
    "process_image",
    "process_image_sync",
    "main",
    # Core algorithms
    "runs_based_detect",
    "edge_aware_detect",
    "quantize_colors",
    "extract_palette",
    "count_colors",
    "detect_optimal_color_count",
    # Downscaling methods
    "downscale_by_dominant_color",
    "downscale_block",
    # Image cleanup
    "alpha_binarization",
    "morphological_cleanup",
    "jaggy_cleaner",
    "finalize_pixels",
    # Advanced features
    "content_adaptive_downscale",
    # Rust-accelerated functions
    "runs_based_detect_accelerated",
    "map_pixels_to_palette_accelerated",
    "WuQuantizerAccelerated",
    "RUST_AVAILABLE",
    # Version
    "__version__",
]
