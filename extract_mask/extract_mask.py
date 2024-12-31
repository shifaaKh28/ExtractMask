import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circle_and_extract(image):
    """
    Detect the circle and extract its parameters from the input image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - extracted_shape (numpy.ndarray): Shape extracted based on circle mask.
    - detected_circle (tuple): Circle parameters (x, y, radius).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 4,
        param1=50,
        param2=30,
        minRadius=gray.shape[0] // 6,
        maxRadius=gray.shape[0] // 2
    )

    if circles is not None:
        # Take the first detected circle
        circles = np.round(circles[0, :]).astype("int")
        circle = circles[0]  # Assuming the largest/most prominent circle

        # Create a mask for the circle
        mask = np.zeros_like(gray)
        cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)

        # Apply the mask to the image to isolate the shape
        extracted_shape = cv2.bitwise_and(gray, gray, mask=mask)
        return extracted_shape, circle

    return None, None

def refine_internal_shape_with_textures(image, circle_params):
    """
    Detect the internal textures and refine the internal shape within the circle.

    Parameters:
    - image (numpy.ndarray): Input image.
    - circle_params (tuple): Detected circle parameters (x, y, radius).

    Returns:
    - refined_texture_shape (numpy.ndarray): Image with the internal textures highlighted.
    """
    if circle_params is None:
        print("No circle detected. Cannot refine internal shape with textures.")
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create a mask for the inner circle area
    circle_mask = np.zeros_like(gray)
    cx, cy, r = circle_params
    cv2.circle(circle_mask, (cx, cy), r - 5, 255, -1)

    # Enhance edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, circle_mask)

    # Find contours and draw them on an empty black image
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_texture_shape = np.zeros_like(image)
    cv2.drawContours(refined_texture_shape, contours, -1, (0, 255, 0), 1)

    return refined_texture_shape

def process_image(image_path):
    """
    Processes the image to detect a circle and refine the internal shape with textures.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - None
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return

    # Step 1: Detect Circle
    detected_shape, detected_circle = detect_circle_and_extract(img)

    if detected_circle is not None:
        # Step 2: Refine Internal Shape Detection with Textures
        refined_texture_shape = refine_internal_shape_with_textures(img, detected_circle)

        # Save the result
        output_path = "refined_texture_shape_output.png"
        cv2.imwrite(output_path, refined_texture_shape)
        print(f"Refined texture shape saved at {output_path}")

        # Display results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Refined Texture Shape")
        plt.imshow(cv2.cvtColor(refined_texture_shape, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("Circle detection failed. Could not process further.")

# Example usage
if __name__ == "__main__":
    image_path = "img3.png"  # Replace with your input image path
    process_image(image_path)
