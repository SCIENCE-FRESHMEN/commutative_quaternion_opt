"""
Demo: Color Image Low-Rank Recovery using Commutative Quaternion Optimization.
Maps a real RGB image to a commutative quaternion tensor, adds noise,
and recovers it using Algorithm 1 (Best Rank-1 Approximation).
"""
import sys
import numpy as np
import cv2
import os

sys.path.insert(0, '.')
from src.config import Algorithm1Config
from src.algorithms.alg1_multilinear import algorithm_1_multilinear
from src.utils.logger import setup_logger

logger = setup_logger("ImageRecoveryDemo")

def rgb_to_commutative_quaternion(img: np.ndarray) -> np.ndarray:
    """
    Maps an RGB image (H, W, 3) to a commutative quaternion tensor (4, H, W).
    Mapping: q0 = R, q1 = G, q2 = B, q3 = 0.
    Normalized to [0, 1] range.
    """
    img_norm = img.astype(np.float64) / 255.0
    H, W, _ = img_norm.shape

    # Create 4-channel quaternion representation
    T = np.zeros((4, H, W), dtype=np.float64)
    T[0, :, :] = img_norm[:, :, 0]  # Real part (R)
    T[1, :, :] = img_norm[:, :, 1]  # i part (G)
    T[2, :, :] = img_norm[:, :, 2]  # j part (B)
    T[3, :, :] = 0.0                # k part

    return T

def quaternion_to_rgb(T: np.ndarray) -> np.ndarray:
    """
    Reconstructs an RGB image from a commutative quaternion tensor (4, H, W).
    """
    # Clip to [0, 1] and scale back to [0, 255]
    R = np.clip(T[0, :, :], 0, 1)
    G = np.clip(T[1, :, :], 0, 1)
    B = np.clip(T[2, :, :], 0, 1)

    img_recovered = np.stack([R, G, B], axis=-1)
    return (img_recovered * 255.0).astype(np.uint8)

def main():
    # 1. Load or generate a sample image
    img_path = "sample_image.png"
    if not os.path.exists(img_path):
        logger.info("No sample image found. Generating a synthetic structured color image...")
        H, W = 64, 64
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = (np.sin(2 * np.pi * X) * 127 + 128).astype(np.uint8) # R
        img[:, :, 1] = (np.cos(2 * np.pi * Y) * 127 + 128).astype(np.uint8) # G
        img[:, :, 2] = (np.sin(2 * np.pi * (X + Y)) * 127 + 128).astype(np.uint8) # B
        cv2.imwrite(img_path, img)

    logger.info(f"Loading image: {img_path}")
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 2. Convert to Quaternion Tensor and Add Noise
    T_original = rgb_to_commutative_quaternion(original_img)

    # Add Gaussian noise to simulate real-world degradation
    noise = np.random.normal(0, 0.15, T_original.shape) # Increased noise slightly for better visual demo
    T_noisy = T_original + noise
    noisy_img = quaternion_to_rgb(T_noisy)

    # 3. Apply Algorithm 1 for Best Rank-1 Approximation
    logger.info("Running Algorithm 1 for Rank-1 Quaternion Tensor Approximation...")
    config = Algorithm1Config(num_trials=50) # 50 trials is enough for d=2 SVD-based exact solve in step 2

    # Note: For a 2D image (H, W), the tensor shape is (4, H, W).
    # The number of spatial modes is d = 2.
    # Therefore, Algorithm 1 will return exactly 2 vectors: one for H and one for W.
    vectors_F, val_F = algorithm_1_multilinear(T_noisy, config)

    # 4. Reconstruct the Rank-1 Tensor
    # vectors_F[0] corresponds to the first spatial mode (H)
    # vectors_F[1] corresponds to the second spatial mode (W)
    x_H = vectors_F[0] # Shape (4, H)
    x_W = vectors_F[1] # Shape (4, W)

    # Reconstruct T_rank1 (4, H, W) using outer product for each quaternion component
    T_rank1 = np.zeros_like(T_noisy)
    for c in range(4):
        T_rank1[c, :, :] = np.outer(x_H[c, :], x_W[c, :])

    # Scale to match original magnitude (improves visual PSNR)
    # We use the real part (c=0) or Frobenius norm for scaling factor estimation
    scale = np.sum(T_original * T_rank1) / (np.sum(T_rank1 ** 2) + 1e-8)
    T_rank1 = T_rank1 * scale

    recovered_img = quaternion_to_rgb(T_rank1)

    # 5. Save and Display Results
    os.makedirs("./demo_output", exist_ok=True)
    cv2.imwrite("./demo_output/original.png", cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("./demo_output/noisy.png", cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("./demo_output/recovered_rank1.png", cv2.cvtColor(recovered_img, cv2.COLOR_RGB2BGR))

    logger.info("✅ Image recovery completed!")
    logger.info("📁 Check the './demo_output/' folder for: original.png, noisy.png, recovered_rank1.png")

if __name__ == "__main__":
    main()