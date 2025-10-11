use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::HashMap;

/// Simple 2x2 matrix for covariance operations
#[derive(Clone, Copy, Debug)]
struct Matrix2x2 {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

impl Matrix2x2 {
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self { a, b, c, d }
    }

    fn multiply(&self, other: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )
    }

    fn determinant(&self) -> f32 {
        self.a * self.d - self.b * self.c
    }

    fn inverse(&self) -> Option<Matrix2x2> {
        let det = self.determinant();
        if det.abs() < 1e-9 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Matrix2x2::new(
            self.d * inv_det,
            -self.b * inv_det,
            -self.c * inv_det,
            self.a * inv_det,
        ))
    }

    // Simple SVD for 2x2 symmetric positive definite matrices
    fn eigenvalues(&self) -> (f32, f32) {
        let trace = self.a + self.d;
        let det = self.determinant();
        let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda1 = (trace + discriminant) * 0.5;
        let lambda2 = (trace - discriminant) * 0.5;
        (lambda1.max(0.0), lambda2.max(0.0))
    }

    fn clamp_eigenvalues(&self, min_val: f32, max_val: f32) -> Matrix2x2 {
        let (mut l1, mut l2) = self.eigenvalues();
        l1 = l1.clamp(min_val, max_val);
        l2 = l2.clamp(min_val, max_val);

        // For symmetric matrices, reconstruct using eigenvalues
        // This is a simplified reconstruction that preserves the matrix structure
        let scale1 = if self.a > 0.0 {
            (l1 / self.a.max(1e-9)).sqrt()
        } else {
            1.0
        };
        let scale2 = if self.d > 0.0 {
            (l2 / self.d.max(1e-9)).sqrt()
        } else {
            1.0
        };

        Matrix2x2::new(
            self.a * scale1,
            self.b * (scale1 + scale2) * 0.5,
            self.c * (scale1 + scale2) * 0.5,
            self.d * scale2,
        )
    }
}

/// Gaussian kernel for content-adaptive downscaling
#[derive(Clone)]
struct GaussianKernel {
    mu: [f32; 2],     // position (x, y)
    sigma: Matrix2x2, // covariance matrix
    nu: [f32; 3],     // color (L, a, b)
}

impl GaussianKernel {
    fn new(x: f32, y: f32, sx: f32, sy: f32) -> Self {
        Self {
            mu: [x, y],
            sigma: Matrix2x2::new(sx, 0.0, 0.0, sy),
            nu: [50.0, 0.0, 0.0], // neutral gray in LAB
        }
    }

    // Accept precomputed inverse to avoid repeated inversion
    fn evaluate_weight(&self, x: f32, y: f32, inv_sigma: &Matrix2x2) -> f32 {
        let dx = x - self.mu[0];
        let dy = y - self.mu[1];
        let exponent = dx * dx * inv_sigma.a + 2.0 * dx * dy * inv_sigma.b + dy * dy * inv_sigma.d;
        (-0.5 * exponent).exp()
    }
}

/// Fast content-adaptive downscaling using EM-C algorithm
pub fn content_adaptive_downscale_rust(
    lab_image: &ArrayView3<f32>,
    target_w: usize,
    target_h: usize,
    num_iterations: usize,
) -> Array3<f32> {
    let (h_in, w_in, _) = lab_image.dim();
    let h_out = target_h;
    let w_out = target_w;
    let rx = w_in as f32 / w_out as f32;
    let ry = h_in as f32 / h_out as f32;

    // Initialize kernels
    let mut kernels: Vec<GaussianKernel> = (0..h_out)
        .flat_map(|yk| {
            (0..w_out).map(move |xk| {
                let center_x = (xk as f32 + 0.5) * rx;
                let center_y = (yk as f32 + 0.5) * ry;
                let initial_sx = (rx / 3.0) * (rx / 3.0);
                let initial_sy = (ry / 3.0) * (ry / 3.0);
                GaussianKernel::new(center_x, center_y, initial_sx, initial_sy)
            })
        })
        .collect();

    let num_kernels = kernels.len();
    let search_radius = 2.0 * rx.max(ry);
    let search_radius_sq = search_radius * search_radius;

    // EM-C iterations
    for iteration in 0..num_iterations {
        // Precompute inverse covariance matrices (safe due to clamping in C-step)
        let inv_sigmas: Vec<Matrix2x2> = kernels
            .par_iter()
            .map(|kernel| kernel.sigma.inverse().unwrap())
            .collect();

        // E-Step: Compute weights in parallel
        let mut all_weights: Vec<Vec<(usize, f32)>> = (0..num_kernels)
            .into_par_iter()
            .map(|k| {
                let kernel = &kernels[k];
                let inv_sigma = &inv_sigmas[k];

                // Determine local pixel bounds for this kernel
                let x_min = (kernel.mu[0] - search_radius).max(0.0) as usize;
                let x_max = (kernel.mu[0] + search_radius).min(w_in as f32 - 1.0) as usize + 1;
                let y_min = (kernel.mu[1] - search_radius).max(0.0) as usize;
                let y_max = (kernel.mu[1] + search_radius).min(h_in as f32 - 1.0) as usize + 1;

                let mut weights = Vec::new();
                for y in y_min..y_max {
                    for x in x_min..x_max {
                        let fx = x as f32;
                        let fy = y as f32;
                        let dx = fx - kernel.mu[0];
                        let dy = fy - kernel.mu[1];
                        let dist_sq = dx * dx + dy * dy;

                        if dist_sq <= search_radius_sq {
                            let weight = kernel.evaluate_weight(fx, fy, inv_sigma);
                            if weight > 1e-5 {
                                weights.push((y * w_in + x, weight));
                            }
                        }
                    }
                }
                weights
            })
            .collect();

        // Normalize weights
        let mut gamma_sums = vec![1e-9f32; h_in * w_in];
        for k in 0..num_kernels {
            let w_sum: f32 = all_weights[k].iter().map(|(_, w)| w).sum::<f32>() + 1e-9;
            for (pixel_idx, weight) in &mut all_weights[k] {
                *weight /= w_sum;
                gamma_sums[*pixel_idx] += *weight;
            }
        }

        // M-Step: Update kernel parameters in parallel
        let new_kernels: Vec<GaussianKernel> = (0..num_kernels)
            .into_par_iter()
            .map(|k| {
                let weights = &all_weights[k];
                let mut w_sum = 1e-9f32;
                let mut new_mu = [0.0f32; 2];
                let mut new_nu = [0.0f32; 3];

                // Update position and color
                for &(pixel_idx, weight) in weights {
                    let y = pixel_idx / w_in;
                    let x = pixel_idx % w_in;
                    let gamma_k_i = weight / gamma_sums[pixel_idx];
                    w_sum += gamma_k_i;

                    new_mu[0] += gamma_k_i * x as f32;
                    new_mu[1] += gamma_k_i * y as f32;
                    new_nu[0] += gamma_k_i * lab_image[(y, x, 0)];
                    new_nu[1] += gamma_k_i * lab_image[(y, x, 1)];
                    new_nu[2] += gamma_k_i * lab_image[(y, x, 2)];
                }

                new_mu[0] /= w_sum;
                new_mu[1] /= w_sum;
                new_nu[0] /= w_sum;
                new_nu[1] /= w_sum;
                new_nu[2] /= w_sum;

                // Update covariance
                let mut cov_sum = [0.0f32; 4]; // [xx, xy, yx, yy]
                for &(pixel_idx, weight) in weights {
                    let y = pixel_idx / w_in;
                    let x = pixel_idx % w_in;
                    let gamma_k_i = weight / gamma_sums[pixel_idx];
                    let dx = x as f32 - new_mu[0];
                    let dy = y as f32 - new_mu[1];

                    cov_sum[0] += gamma_k_i * dx * dx;
                    cov_sum[1] += gamma_k_i * dx * dy;
                    cov_sum[3] += gamma_k_i * dy * dy;
                }

                cov_sum[0] /= w_sum;
                cov_sum[1] /= w_sum;
                cov_sum[2] = cov_sum[1]; // symmetric
                cov_sum[3] /= w_sum;

                let new_sigma = Matrix2x2::new(cov_sum[0], cov_sum[1], cov_sum[2], cov_sum[3]);

                GaussianKernel {
                    mu: new_mu,
                    sigma: new_sigma,
                    nu: new_nu,
                }
            })
            .collect();

        // C-Step: Clamp kernel sizes for sharpness
        kernels = new_kernels
            .into_iter()
            .map(|mut kernel| {
                kernel.sigma = kernel.sigma.clamp_eigenvalues(0.05, 0.1);
                kernel
            })
            .collect();
    }

    // Generate output image
    let mut output = Array3::<f32>::zeros((h_out, w_out, 3));
    for yk in 0..h_out {
        for xk in 0..w_out {
            let k_idx = yk * w_out + xk;
            if k_idx < kernels.len() {
                for c in 0..3 {
                    output[(yk, xk, c)] = kernels[k_idx].nu[c];
                }
            }
        }
    }

    output
}

/// Downscale image using mode (most frequent color) method
pub fn downscale_mode(image: &ArrayView3<u8>, scale: usize) -> Array3<u8> {
    let (height, width, channels) = image.dim();
    let target_h = height / scale;
    let target_w = width / scale;
    let has_alpha = channels == 4;

    // Collect results in parallel
    let results: Vec<_> = (0..target_h)
        .into_par_iter()
        .flat_map(|ty| {
            (0..target_w)
                .into_par_iter()
                .map(move |tx| {
                    let mut color_counts = HashMap::with_capacity(scale * scale);
                    let mut alpha_values = Vec::with_capacity(scale * scale);

                    // Scan the scale x scale block
                    let y_start = ty * scale;
                    let x_start = tx * scale;
                    let y_end = ((ty + 1) * scale).min(height);
                    let x_end = ((tx + 1) * scale).min(width);

                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            if has_alpha {
                                let alpha = image[(y, x, 3)];
                                alpha_values.push(alpha);

                                if alpha >= 128 {
                                    // Pack RGB into 24-bit integer for fast hashing
                                    let r = image[(y, x, 0)] as u32;
                                    let g = image[(y, x, 1)] as u32;
                                    let b = image[(y, x, 2)] as u32;
                                    let color_key = (r << 16) | (g << 8) | b;
                                    *color_counts.entry(color_key).or_insert(0) += 1;
                                }
                            } else {
                                let r = image[(y, x, 0)] as u32;
                                let g = image[(y, x, 1)] as u32;
                                let b = image[(y, x, 2)] as u32;
                                let color_key = (r << 16) | (g << 8) | b;
                                *color_counts.entry(color_key).or_insert(0) += 1;
                            }
                        }
                    }

                    // Calculate output pixel
                    let mut pixel = vec![0u8; channels];

                    if has_alpha && !alpha_values.is_empty() {
                        // Calculate median alpha efficiently
                        alpha_values.sort_unstable();
                        let median_alpha = alpha_values[alpha_values.len() / 2];
                        let final_alpha = if median_alpha >= 128 { 255 } else { 0 };

                        pixel[3] = final_alpha;

                        if final_alpha == 0 {
                            // Transparent pixel - set to black
                            pixel[0] = 0;
                            pixel[1] = 0;
                            pixel[2] = 0;
                            return (ty, tx, pixel);
                        }
                    }

                    if !color_counts.is_empty() {
                        // Find most frequent color (mode)
                        let most_frequent_color = color_counts
                            .iter()
                            .max_by_key(|(_, &count)| count)
                            .map(|(&color, _)| color)
                            .unwrap_or(0);

                        pixel[0] = ((most_frequent_color >> 16) & 0xFF) as u8;
                        pixel[1] = ((most_frequent_color >> 8) & 0xFF) as u8;
                        pixel[2] = (most_frequent_color & 0xFF) as u8;

                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    } else {
                        // No pixels - set to black
                        pixel[0] = 0;
                        pixel[1] = 0;
                        pixel[2] = 0;
                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    }

                    (ty, tx, pixel)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build output array from results
    let mut output = Array3::<u8>::zeros((target_h, target_w, channels));
    for (ty, tx, pixel) in results {
        for (c, &value) in pixel.iter().enumerate() {
            output[(ty, tx, c)] = value;
        }
    }

    output
}

/// Downscale image using dominant color method
pub fn downscale_dominant(image: &ArrayView3<u8>, scale: usize, threshold: f32) -> Array3<u8> {
    let (height, width, channels) = image.dim();
    let target_h = height / scale;
    let target_w = width / scale;
    let has_alpha = channels == 4;

    // Collect results in parallel
    let results: Vec<_> = (0..target_h)
        .into_par_iter()
        .flat_map(|ty| {
            (0..target_w)
                .into_par_iter()
                .map(move |tx| {
                    let mut color_counts = HashMap::with_capacity(scale * scale);
                    let mut alpha_values = Vec::with_capacity(scale * scale);
                    let mut total_opaque = 0u32;

                    // Scan the scale x scale block
                    let y_start = ty * scale;
                    let x_start = tx * scale;
                    let y_end = ((ty + 1) * scale).min(height);
                    let x_end = ((tx + 1) * scale).min(width);

                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            if has_alpha {
                                let alpha = image[(y, x, 3)];
                                alpha_values.push(alpha);

                                if alpha >= 128 {
                                    // Pack RGB into 24-bit integer (same as JavaScript)
                                    let r = image[(y, x, 0)] as u32;
                                    let g = image[(y, x, 1)] as u32;
                                    let b = image[(y, x, 2)] as u32;
                                    let color_key = (r << 16) | (g << 8) | b;

                                    *color_counts.entry(color_key).or_insert(0) += 1;
                                    total_opaque += 1;
                                }
                            } else {
                                let r = image[(y, x, 0)] as u32;
                                let g = image[(y, x, 1)] as u32;
                                let b = image[(y, x, 2)] as u32;
                                let color_key = (r << 16) | (g << 8) | b;

                                *color_counts.entry(color_key).or_insert(0) += 1;
                                total_opaque += 1;
                            }
                        }
                    }

                    // Calculate output pixel
                    let mut pixel = vec![0u8; channels];

                    if has_alpha && !alpha_values.is_empty() {
                        // Calculate median alpha efficiently
                        alpha_values.sort_unstable();
                        let median_alpha = alpha_values[alpha_values.len() / 2];
                        let final_alpha = if median_alpha >= 128 { 255 } else { 0 };

                        pixel[3] = final_alpha;

                        if final_alpha == 0 {
                            // Transparent pixel - set to black
                            pixel[0] = 0;
                            pixel[1] = 0;
                            pixel[2] = 0;
                            return (ty, tx, pixel);
                        }
                    }

                    if total_opaque > 0 {
                        // Find dominant color
                        let (dominant_color, max_count) = color_counts
                            .iter()
                            .max_by_key(|(_, &count)| count)
                            .map(|(&color, &count)| (color, count))
                            .unwrap_or((0, 0));

                        let threshold_count = (total_opaque as f32 * threshold) as u32;

                        if max_count >= threshold_count {
                            // Use dominant color
                            pixel[0] = ((dominant_color >> 16) & 0xFF) as u8;
                            pixel[1] = ((dominant_color >> 8) & 0xFF) as u8;
                            pixel[2] = (dominant_color & 0xFF) as u8;
                        } else {
                            // Calculate mean color
                            let mut r_sum = 0u64;
                            let mut g_sum = 0u64;
                            let mut b_sum = 0u64;

                            for (&color, &count) in &color_counts {
                                let count = count as u64;
                                r_sum += ((color >> 16) & 0xFF) as u64 * count;
                                g_sum += ((color >> 8) & 0xFF) as u64 * count;
                                b_sum += (color & 0xFF) as u64 * count;
                            }

                            pixel[0] = (r_sum / total_opaque as u64) as u8;
                            pixel[1] = (g_sum / total_opaque as u64) as u8;
                            pixel[2] = (b_sum / total_opaque as u64) as u8;
                        }

                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    } else {
                        // No opaque pixels - set to black
                        pixel[0] = 0;
                        pixel[1] = 0;
                        pixel[2] = 0;
                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    }

                    (ty, tx, pixel)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build output array from results
    let mut output = Array3::<u8>::zeros((target_h, target_w, channels));
    for (ty, tx, pixel) in results {
        for (c, &value) in pixel.iter().enumerate() {
            output[(ty, tx, c)] = value;
        }
    }

    output
}

/// Make background transparent by flood-filling from specified starting points
pub fn make_background_transparent_rust(
    image: &ArrayView3<u8>,
    tolerance: i32,
    mode: &str,
) -> Array3<u8> {
    let (height, width, channels) = image.dim();

    // Ensure image has alpha channel
    let (h, w, has_alpha) = if channels == 3 {
        // Need to add alpha channel
        (height, width, false)
    } else {
        (height, width, true)
    };

    // Create result with alpha channel
    let mut result = if !has_alpha {
        // Convert RGB to RGBA
        let mut rgba = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                rgba[(y, x, 0)] = image[(y, x, 0)];
                rgba[(y, x, 1)] = image[(y, x, 1)];
                rgba[(y, x, 2)] = image[(y, x, 2)];
                rgba[(y, x, 3)] = 255; // Fully opaque
            }
        }
        rgba
    } else {
        // Copy existing RGBA
        let mut rgba = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                rgba[(y, x, 0)] = image[(y, x, 0)];
                rgba[(y, x, 1)] = image[(y, x, 1)];
                rgba[(y, x, 2)] = image[(y, x, 2)];
                rgba[(y, x, 3)] = image[(y, x, 3)];
            }
        }
        rgba
    };

    let mut visited = vec![vec![false; w]; h];

    // Helper function to check if two colors are similar
    let is_similar = |c1: [u8; 3], c2: [u8; 3], tol: i32| -> bool {
        ((c1[0] as i32 - c2[0] as i32).abs() <= tol)
            && ((c1[1] as i32 - c2[1] as i32).abs() <= tol)
            && ((c1[2] as i32 - c2[2] as i32).abs() <= tol)
    };

    // Flood fill function using iterative stack-based approach
    let flood_fill = |result: &mut Array3<u8>,
                      visited: &mut Vec<Vec<bool>>,
                      start_y: usize,
                      start_x: usize,
                      tol: i32| {
        // Skip if already visited or transparent
        if visited[start_y][start_x] || result[(start_y, start_x, 3)] < 128 {
            return;
        }

        let target_color = [
            result[(start_y, start_x, 0)],
            result[(start_y, start_x, 1)],
            result[(start_y, start_x, 2)],
        ];

        let mut stack = vec![(start_y, start_x)];

        while let Some((y, x)) = stack.pop() {
            // Bounds check
            if y >= h || x >= w {
                continue;
            }

            // Skip if already visited
            if visited[y][x] {
                continue;
            }

            // Skip if already transparent
            if result[(y, x, 3)] < 128 {
                visited[y][x] = true;
                continue;
            }

            // Check if color matches
            let current_color = [result[(y, x, 0)], result[(y, x, 1)], result[(y, x, 2)]];
            if !is_similar(current_color, target_color, tol) {
                continue;
            }

            // Mark as visited and make transparent
            visited[y][x] = true;
            result[(y, x, 0)] = 0;
            result[(y, x, 1)] = 0;
            result[(y, x, 2)] = 0;
            result[(y, x, 3)] = 0;

            // Add neighbors to stack (with bounds checking)
            if y > 0 {
                stack.push((y - 1, x)); // up
            }
            if y + 1 < h {
                stack.push((y + 1, x)); // down
            }
            if x > 0 {
                stack.push((y, x - 1)); // left
            }
            if x + 1 < w {
                stack.push((y, x + 1)); // right
            }
        }
    };

    // Flood fill based on mode
    if mode == "corners" {
        // Only flood fill from the four corners
        if !visited[0][0] {
            flood_fill(&mut result, &mut visited, 0, 0, tolerance);
        }
        if !visited[0][w - 1] {
            flood_fill(&mut result, &mut visited, 0, w - 1, tolerance);
        }
        if !visited[h - 1][0] {
            flood_fill(&mut result, &mut visited, h - 1, 0, tolerance);
        }
        if !visited[h - 1][w - 1] {
            flood_fill(&mut result, &mut visited, h - 1, w - 1, tolerance);
        }
    } else if mode == "midpoints" {
        // Flood fill from the midpoint of each edge
        let mid_x = w / 2;
        let mid_y = h / 2;

        // Top edge midpoint
        if !visited[0][mid_x] {
            flood_fill(&mut result, &mut visited, 0, mid_x, tolerance);
        }
        // Bottom edge midpoint
        if !visited[h - 1][mid_x] {
            flood_fill(&mut result, &mut visited, h - 1, mid_x, tolerance);
        }
        // Left edge midpoint
        if !visited[mid_y][0] {
            flood_fill(&mut result, &mut visited, mid_y, 0, tolerance);
        }
        // Right edge midpoint
        if !visited[mid_y][w - 1] {
            flood_fill(&mut result, &mut visited, mid_y, w - 1, tolerance);
        }
    } else {
        // mode == "edges" (default)
        // Flood fill from all edges
        // Top and bottom edges
        for x in 0..w {
            if !visited[0][x] {
                flood_fill(&mut result, &mut visited, 0, x, tolerance);
            }
            if !visited[h - 1][x] {
                flood_fill(&mut result, &mut visited, h - 1, x, tolerance);
            }
        }

        // Left and right edges
        for y in 0..h {
            if !visited[y][0] {
                flood_fill(&mut result, &mut visited, y, 0, tolerance);
            }
            if !visited[y][w - 1] {
                flood_fill(&mut result, &mut visited, y, w - 1, tolerance);
            }
        }
    }

    result
}
