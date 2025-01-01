import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_inner_shape_segmentation(image_path):
    """
    Enhances the segmentation clarity of the inner shape of a specimen.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        dilated_edges: Enhanced edges after applying dilation.
        mask: Refined mask for the largest contour.
        segmented_inner_shape: Segmented inner shape of the specimen.
    """
    # Step 1: Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    
    # Step 3: Smooth the image
    smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    
    # Step 4: Detect edges using Canny
    edges = cv2.Canny(smoothed_image, 50, 150)
    
    # Step 5: Apply dilation to strengthen the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 6: Detect contours and focus on the largest
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)
    
    # Step 7: Apply the mask to isolate the inner shape
    segmented_inner_shape = cv2.bitwise_and(image, image, mask=mask)
    
    return dilated_edges, mask, segmented_inner_shape

# Example usage with visualization
def process_and_display_segmentation(image_path):
    """
    Processes the input image using the enhanced segmentation method
    and visualizes the results.

    Args:
        image_path (str): Path to the input image.
    """
    # Perform enhanced segmentation
    dilated_edges, mask, segmented_inner_shape = enhance_inner_shape_segmentation(image_path)
    
    # Display results
    plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.axis('off')

    # Dilated edges
    plt.subplot(2, 3, 2)
    plt.title("Dilated Edges")
    plt.imshow(dilated_edges, cmap='gray')
    plt.axis('off')

    # Enhanced mask
    plt.subplot(2, 3, 3)
    plt.title("Enhanced Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # Segmented inner shape (enhanced)
    plt.subplot(2, 3, 4)
    plt.title("Segmented Inner Shape (Enhanced)")
    plt.imshow(segmented_inner_shape, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'path_to_your_image.tif'  # Replace with your image path
process_and_display_segmentation(image_path)
