import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os

# Create a sample image if no image exists, or load an existing one
def create_sample_image(width=200, height=200):
    """Create a sample image with gradients and patterns"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = int(255 * i / height)  # Red channel
            img[i, j, 1] = int(255 * j / width)   # Green channel
            img[i, j, 2] = int(255 * (1 - i/height))  # Blue channel
    
    # Add some patterns
    for i in range(50, 150):
        for j in range(50, 150):
            img[i, j, :] = [255, 100, 50]
    
    return img

def load_or_create_image(filepath='sample_image.jpg', width=200, height=200):
    """Load image from file or create a sample one"""
    if os.path.exists(filepath):
        print(f"Loading image from {filepath}")
        img = Image.open(filepath).convert('RGB')
        img = np.array(img)
    else:
        print("Creating sample image")
        img = create_sample_image(width, height)
        # Save the sample image
        Image.fromarray(img).save('sample_image.jpg')
    
    return img

# Load or create image
print("=" * 60)
print("PCA for Image Compression")
print("=" * 60)
original_image = load_or_create_image()
print(f"Original image shape: {original_image.shape}")
print()

# Function to compress image using PCA
def compress_image_pca(image, n_components):
    """Compress image using PCA"""
    height, width, channels = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)
    
    # Process each color channel separately
    for c in range(channels):
        # Reshape the channel to 2D for PCA
        channel_data = image[:, :, c].astype(np.float32)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, height, width))
        
        # Reshape to 2D (height, width) for fitting
        channel_2d = channel_data.reshape(height, width)
        
        # Flatten to apply PCA
        channel_flat = channel_2d.reshape(height, -1)
        
        # Fit and transform
        channel_pca = pca.fit_transform(channel_flat)
        
        # Reconstruct
        channel_reconstructed = pca.inverse_transform(channel_pca)
        
        # Store reconstructed channel
        compressed_image[:, :, c] = channel_reconstructed
    
    return np.clip(compressed_image, 0, 255).astype(np.uint8), channel_pca, pca

# Compress image with different number of components
print("=" * 60)
print("Compression Analysis:")
print("=" * 60)

components_list = [5, 10, 20, 50]
compressed_images = {}
mse_values = {}
compression_ratios = {}

height, width, channels = original_image.shape
original_size = height * width * channels * 8  # bits (8 bits per pixel)

for n_comp in components_list:
    print(f"\nCompressing with {n_comp} principal components...")
    
    # Compress
    compressed_img, pca_transformed, pca_model = compress_image_pca(original_image, n_comp)
    compressed_images[n_comp] = compressed_img
    
    # Calculate MSE
    mse = np.mean((original_image.astype(float) - compressed_img.astype(float)) ** 2)
    mse_values[n_comp] = mse
    
    # Calculate compression ratio
    # Compressed size = (height * n_comp + n_comp * width) * 3 channels (simplified)
    compressed_size = (height * n_comp + n_comp * width) * channels * 8
    compression_ratio = original_size / compressed_size
    compression_ratios[n_comp] = compression_ratio
    
    print(f"  MSE: {mse:.4f}")
    print(f"  Compression Ratio: {compression_ratio:.2f}x")
    print(f"  Space Savings: {(1 - 1/compression_ratio) * 100:.2f}%")

print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Compressed images
comp_indices = [0, 1, 2]
for idx, n_comp in enumerate(components_list[:3]):
    ax = axes[0, idx + 1] if idx == 0 else axes[1, idx - 1]
    ax.imshow(compressed_images[n_comp])
    ax.set_title(f'{n_comp} Components\nMSE: {mse_values[n_comp]:.2f}, Ratio: {compression_ratios[n_comp]:.2f}x')
    ax.axis('off')

# Hide the extra subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('pca_image_compression.png', dpi=100, bbox_inches='tight')
print("Saved comparison plot as 'pca_image_compression.png'")

# Plot MSE vs Components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mse_list = [mse_values[n] for n in components_list]
ratio_list = [compression_ratios[n] for n in components_list]

ax1.plot(components_list, mse_list, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Principal Components')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.set_title('Reconstruction Error vs Components')
ax1.grid(True, alpha=0.3)

ax2.plot(components_list, ratio_list, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Compression Ratio')
ax2.set_title('Compression Ratio vs Components')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_compression_analysis.png', dpi=100, bbox_inches='tight')
print("Saved analysis plot as 'pca_compression_analysis.png'")

print("\n" + "=" * 60)
print("Image Compression Summary:")
print("=" * 60)
print(f"Original image size: {original_size} bits ({original_size/8/1024:.2f} KB)")
print()
for n_comp in components_list:
    compressed_size = (height * n_comp + n_comp * width) * channels * 8
    print(f"{n_comp:2d} components: {compressed_size} bits ({compressed_size/8/1024:.2f} KB) - "
          f"Ratio: {compression_ratios[n_comp]:.2f}x, MSE: {mse_values[n_comp]:.2f}")

plt.show()
