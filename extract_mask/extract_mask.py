import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_circle(image):
    """
    Detect and visualize only the circle in the image
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    tuple: (processed_image, circle_parameters)
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(gray.shape),
        param1=50,
        param2=30,
        minRadius=min(gray.shape) // 3,
        maxRadius=min(gray.shape) // 2
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = circles[0]
        
        # Draw only the circle
        circle_visualization = image_rgb.copy()
        cv2.circle(circle_visualization,
                  center=(largest_circle[0], largest_circle[1]),
                  radius=largest_circle[2] + 5,
                  color=(255, 0, 0),  # Red
                  thickness=2)
        
        return circle_visualization, largest_circle
    
    return image_rgb, None
def detect_internal_shape(image, circle_params):
    """
    Detect and visualize the shape inside the detected circle using Canny edge detection.
    
    Parameters:
    image (numpy.ndarray): Input image
    circle_params: Parameters of the detected circle (center_x, center_y, radius)
    
    Returns:
    tuple: (processed_image, shape_name, internal_mask)
    """
    if circle_params is None:
        return image, "no circle detected", None

    image_rgb = image.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Create circular mask
    circle_mask = np.zeros_like(gray)
    cx, cy, r = circle_params
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)

    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, circle_mask)

    # Filter contours
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    internal_mask = np.zeros_like(masked_edges)

    for cnt in contours:
        # Check if the contour's points are entirely within a smaller circle (excludes the boundary)
        inside = True
        for point in cnt:
            x, y = point[0]
            distance_to_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance_to_center >= r - 10:  # Allow a margin near the edge
                inside = False
                break
        if inside:
            cv2.drawContours(internal_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Visualize the internal contours
    shape_visualization = image_rgb.copy()
    shape_visualization[internal_mask != 0] = (0, 255, 0)  # Green edges

    return shape_visualization, "internal shape", internal_mask



def process_image_in_steps(image_path):
    """
    Process a single image showing each step separately
    
    Parameters:
    image_path (str): Path to the input image
    
    Returns:
    None
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    
    # Step 1: Detect circle
    circle_image, circle_params = detect_circle(img)
    
    # Step 2: Detect and color internal shape
    shape_image, _, internal_mask = detect_internal_shape(img, circle_params)
    
    # Step 3: Final result with only the internal green contour
    final_image = np.zeros_like(img)
    if internal_mask is not None:
        final_image[internal_mask != 0] = (0, 255, 0)  # Green contour only
    
    # Combine steps into one visualization
    combined_image = np.hstack([circle_image, shape_image, final_image])
    
    # Add titles to the steps
    height, width, _ = combined_image.shape
    title_bar = np.zeros((50, width, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(title_bar, "Step 1: Circle Detection", (50, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(title_bar, "Step 2: Shape Detection", (width // 3 + 50, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(title_bar, "Step 3: Final Result", (2 * width // 3 + 50, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    final_combined = np.vstack([title_bar, combined_image])
    
    # Save the combined result
    output_path = "combined_result.jpg"
    cv2.imwrite(output_path, final_combined)
    print(f"Saved combined result to {output_path}")

if __name__ == "__main__":
    image_path = "img1.png"  # Path to your single image
    process_image_in_steps(image_path)
