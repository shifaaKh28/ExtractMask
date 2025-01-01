import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    
    # Step 2: Smooth the image to reduce noise
    smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    
    # Step 3: Apply Canny edge detection
    edges = cv2.Canny(smoothed_image, 50, 150)
    
    # Step 4: Detect contours and find the largest circular boundary
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Create a mask for the largest contour
    mask = np.zeros_like(image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Fit a circle around the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask, center, radius, 255, thickness=-1)  # Create circular mask
    
    # Step 6: Refine the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    refined_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    # Step 7: Apply the mask to isolate the circular region
    isolated_circular_region = cv2.bitwise_and(image, image, mask=refined_mask)
    
    # Step 8: Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Refined Mask")
    plt.imshow(refined_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Isolated Circular Region")
    plt.imshow(isolated_circular_region, cmap='gray')
    plt.axis('off')
    
    plt.show()

# Example Usage
# Replace 'your_image_path.png' with the path to your image
process_image('your_image_path.png')
