import cv2
import numpy as np
import argparse


def canny_edge_detection(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Create a white background
    white_bg = np.ones_like(img) * 255

    # Invert the edges to get black lines on white background
    inv_edges = cv2.bitwise_not(edges)

    # Combine the white background with the inverted edges
    result = cv2.bitwise_and(white_bg, white_bg, mask=inv_edges)

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Processed image saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canny Edge Detection")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the output image")
    args = parser.parse_args()

    canny_edge_detection(args.input_image, args.output_image)
