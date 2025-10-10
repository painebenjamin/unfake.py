"""
content_adaptive.py - Content-Adaptive Image Downscaling
Based on the paper "Content-Adaptive Image Downscaling" by Kopf et al.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("unfake.py")


def multiply_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two 2x2 matrices"""
    return np.array(  # type: ignore[no-any-return]
        [
            [a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0], a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]],
            [a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0], a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]],
        ]
    )


def content_adaptive_core(src_lab: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Core content-adaptive downscaling algorithm in LAB color space

    Args:
        src_lab: Source image in LAB color space (32-bit float)
        target_w: Target width
        target_h: Target height

    Returns:
        Downscaled image in LAB color space
    """
    logger.warning("Content-adaptive downscaling is computationally intensive")

    NUM_ITERATIONS = 5
    h_in, w_in = src_lab.shape[:2]
    h_out, w_out = target_h, target_w
    rx = w_in / w_out
    ry = h_in / h_out

    # Split LAB channels
    L_plane = src_lab[:, :, 0]
    a_plane = src_lab[:, :, 1]
    b_plane = src_lab[:, :, 2]

    # Initialize kernels
    mu_k = []  # Kernel positions
    sigma_k = []  # Kernel covariances
    nu_k = []  # Kernel colors

    for yk in range(h_out):
        for xk in range(w_out):
            k_idx = yk * w_out + xk
            # Initialize position at center of corresponding input region
            mu_k.append([(xk + 0.5) * rx, (yk + 0.5) * ry])
            # Initialize covariance (small kernels for sharpness)
            initial_sx = (rx / 3) * (rx / 3)
            initial_sy = (ry / 3) * (ry / 3)
            sigma_k.append([initial_sx, 0, 0, initial_sy])
            # Initialize with neutral gray color
            nu_k.append([50.0, 0.0, 0.0])

    # EM-C iterations
    for iteration in range(NUM_ITERATIONS):
        logger.info(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")

        # E-Step: Compute weights
        gamma_sum_per_pixel = np.zeros((h_in, w_in)) + 1e-9
        w_ki: list[dict[int, float]] = [{} for _ in range(w_out * h_out)]

        for k in range(w_out * h_out):
            s0, s1, s2, s3 = sigma_k[k]
            det = s0 * s3 - s1 * s2
            inv_det = 1.0 / (det + 1e-9)
            sigma_inv = [s3 * inv_det, -s1 * inv_det, -s2 * inv_det, s0 * inv_det]

            mu_x, mu_y = mu_k[k]
            # Limit search region for efficiency
            i_min_x = max(0, int(mu_x - 2 * rx))
            i_max_x = min(w_in, int(mu_x + 2 * rx) + 1)
            i_min_y = max(0, int(mu_y - 2 * ry))
            i_max_y = min(h_in, int(mu_y + 2 * ry) + 1)

            w_sum = 1e-9
            for yi in range(i_min_y, i_max_y):
                for xi in range(i_min_x, i_max_x):
                    dx = xi - mu_x
                    dy = yi - mu_y
                    exponent = (
                        dx * dx * sigma_inv[0] + 2 * dx * dy * sigma_inv[1] + dy * dy * sigma_inv[3]
                    )
                    weight = np.exp(-0.5 * exponent)

                    if weight > 1e-5:
                        i = yi * w_in + xi
                        w_ki[k][i] = weight
                        w_sum += weight

            # Normalize weights
            for i, weight in w_ki[k].items():
                normalized_w = weight / w_sum
                w_ki[k][i] = normalized_w
                yi = i // w_in
                xi = i % w_in
                gamma_sum_per_pixel[yi, xi] += normalized_w

        # M-Step: Update kernel parameters
        new_mu_k = []
        new_sigma_k = []
        new_nu_k = []

        for k in range(w_out * h_out):
            w_sum = 1e-9
            new_mu: list[float] = [0.0, 0.0]
            new_nu: list[float] = [0.0, 0.0, 0.0]

            for i, wk in w_ki[k].items():
                yi = i // w_in
                xi = i % w_in
                gamma_k_i = wk / gamma_sum_per_pixel[yi, xi]
                w_sum += gamma_k_i

                new_mu[0] += gamma_k_i * xi
                new_mu[1] += gamma_k_i * yi
                new_nu[0] += gamma_k_i * L_plane[yi, xi]
                new_nu[1] += gamma_k_i * a_plane[yi, xi]
                new_nu[2] += gamma_k_i * b_plane[yi, xi]

            new_mu[0] /= w_sum
            new_mu[1] /= w_sum
            new_nu[0] /= w_sum
            new_nu[1] /= w_sum
            new_nu[2] /= w_sum

            new_mu_k.append(new_mu)
            new_nu_k.append(new_nu)

            # Update covariance
            new_sigma: list[float] = [0.0, 0.0, 0.0, 0.0]
            for i, wk in w_ki[k].items():
                yi = i // w_in
                xi = i % w_in
                gamma_k_i = wk / gamma_sum_per_pixel[yi, xi]
                dx = xi - new_mu[0]
                dy = yi - new_mu[1]

                new_sigma[0] += gamma_k_i * dx * dx
                new_sigma[1] += gamma_k_i * dx * dy
                new_sigma[3] += gamma_k_i * dy * dy

            new_sigma[0] /= w_sum
            new_sigma[1] /= w_sum
            new_sigma[2] = new_sigma[1]  # Symmetric
            new_sigma[3] /= w_sum

            new_sigma_k.append(new_sigma)

        # C-Step: Clamp kernel sizes for sharpness
        for k in range(w_out * h_out):
            sigma_arr = new_sigma_k[k]
            sigma_mat = np.array([[sigma_arr[0], sigma_arr[1]], [sigma_arr[2], sigma_arr[3]]])

            # SVD decomposition
            u, s, vt = np.linalg.svd(sigma_mat)

            # Clamp eigenvalues to maintain sharpness
            # These values prevent kernels from becoming too large
            s[0] = max(0.05, min(s[0], 0.1))
            s[1] = max(0.05, min(s[1], 0.1))

            # Reconstruct matrix
            s_diag = np.diag(s)
            new_sigma_mat = multiply_2x2(multiply_2x2(u, s_diag), vt)

            final_sigma = [
                new_sigma_mat[0, 0],
                new_sigma_mat[0, 1],
                new_sigma_mat[1, 0],
                new_sigma_mat[1, 1],
            ]

            mu_k[k] = new_mu_k[k]
            sigma_k[k] = final_sigma
            nu_k[k] = new_nu_k[k]

    # Construct output image from kernel colors
    out_lab = np.zeros((h_out, w_out, 3), dtype=np.float32)
    for yk in range(h_out):
        for xk in range(w_out):
            k_idx = yk * w_out + xk
            out_lab[yk, xk] = nu_k[k_idx]

    return out_lab  # type: ignore[no-any-return]


def content_adaptive_downscale(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    EXPERIMENTAL: High-quality content-adaptive downscaling

    Args:
        image: Input image (RGBA or RGB)
        target_w: Target width
        target_h: Target height

    Returns:
        Downscaled image
    """
    logger.info("Using content-adaptive downscaling with Rust acceleration")

    h, w = image.shape[:2]
    has_alpha = image.shape[2] == 4

    # Separate alpha channel if present
    if has_alpha:
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
    else:
        rgb = image

    # Convert to LAB color space for processing
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2Lab)

    # Apply content-adaptive algorithm using Rust acceleration
    from .pixel_rust_integration import content_adaptive_downscale_accelerated

    out_lab = content_adaptive_downscale_accelerated(lab, target_w, target_h)

    # Convert back to RGB
    out_rgb_float = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
    out_rgb = np.clip(out_rgb_float * 255, 0, 255).astype(np.uint8)

    # Handle alpha channel separately
    if has_alpha:
        # Use area interpolation for alpha (best for shrinking)
        out_alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Combine channels
        out_image = np.dstack([out_rgb, out_alpha])
    else:
        out_image = out_rgb

    return out_image  # type: ignore[no-any-return]
