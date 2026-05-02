import cv2
import numpy as np
from sklearn.cluster import KMeans

def auto_find_rot_hsv(image, k=3):
    """
    Use Unsupervised Machine Learning (K-Means Clustering) to segment
    and automatically find the HSV color range of the rotten areas.
    
    Args:
        image: BGR image (background already removed).
        k: Number of color clusters.
        
    Returns:
        lower_rot, upper_rot: numpy arrays for HSV lower/upper bounds.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Filter out absolute black pixels (background) by requiring Value > 10
    mask = hsv_image[:, :, 2] > 10
    pixels = hsv_image[mask].reshape(-1, 3)
    
    if len(pixels) == 0:
        return np.array([0, 0, 0]), np.array([179, 255, 255])
        
    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    # Defective parts typically have the lowest brightness (Value) channel
    rot_cluster_idx = np.argmin(centers[:, 2])
    rot_pixels = pixels[labels == rot_cluster_idx]
    
    # Calculate boundary (min, max) for the defective cluster
    lower_rot = np.min(rot_pixels, axis=0)
    upper_rot = np.max(rot_pixels, axis=0)
    
    # Clip and return properly formatted bounds
    lower_rot = np.clip(lower_rot, 0, [179, 255, 255]).astype(np.uint8)
    upper_rot = np.clip(upper_rot, 0, [179, 255, 255]).astype(np.uint8)
    
    return lower_rot, upper_rot
