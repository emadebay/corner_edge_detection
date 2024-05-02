#corner detection
import cv2
import numpy as np
from scipy.signal import convolve2d

def get_image_filter():

    horizontal_filter =  np.array([
        [-1,0,-1],
        [-2,0,2],
        [-1,0,1]
    ])

    vertical_filter = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])
    return horizontal_filter, vertical_filter

def calculate_the_second_moment_matrix(horizontal_derivatives, vertical_derivatives):
    # Calculate the elements of the second moment matrix
    Ixx = horizontal_derivatives ** 2
    Ixy = horizontal_derivatives * vertical_derivatives
    Iyy = vertical_derivatives ** 2
    
    # Return the second moment matrix
    print(Ixx.shape, Ixy.shape, Iyy.shape)
    return Ixx, Ixy, Iyy

def calculate_corner_response(image, Ixx, Ixy, Iyy, k=0.04):
    # Initialize the corner response matrix with zeros
    R = np.zeros_like(image, dtype=np.float32)
    
    # Pad the second moment matrices to handle edge cases
    padded_Ixx = np.pad(Ixx, 1, mode='edge')
    padded_Ixy = np.pad(Ixy, 1, mode='edge')
    padded_Iyy = np.pad(Iyy, 1, mode='edge')
    
    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the corresponding elements from the padded second moment matrices
            window_Ixx = padded_Ixx[i:i+3, j:j+3]
            window_Ixy = padded_Ixy[i:i+3, j:j+3]
            window_Iyy = padded_Iyy[i:i+3, j:j+3]
            
            # Compute the elements of the matrix H
            H = np.array([[np.sum(window_Ixx), np.sum(window_Ixy)],
                          [np.sum(window_Ixy), np.sum(window_Iyy)]])
            
            # Compute the determinant and trace of H
            det_M = np.linalg.det(H)
            trace_M = np.trace(H)
            
            # Compute the corner response using Harris corner detection method
            R[i, j] = det_M - k * trace_M ** 2
    return R



def non_max_suppression(corners, corner_response, window_size=3):
    suppressed_corners = []

    # Iterate over all detected corners
    for corner in corners:
        y, x = corner

        # Define the region of interest around the current corner
        x_min = max(0, x - window_size // 2)
        x_max = min(corner_response.shape[1] - 1, x + window_size // 2)
        y_min = max(0, y - window_size // 2)
        y_max = min(corner_response.shape[0] - 1, y + window_size // 2)

        # Extract the neighborhood around the current corner
        neighborhood = corner_response[y_min:y_max+1, x_min:x_max+1]

        # Check if the corner response at the current pixel is the maximum within its neighborhood
        if corner_response[y, x] == np.max(neighborhood):
            suppressed_corners.append(corner)

    return suppressed_corners

def detect_corners_binary(image_path, threshold=0.1):
    # Load the image from the file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Compute image derivatives
    horizontal_filter, vertical_filter = get_image_filter()
    horizontal_derivatives = convolve2d(image, horizontal_filter, mode='same')
    vertical_derivatives = convolve2d(image, vertical_filter, mode='same')
    
    # Compute second moment matrix
    Ixx, Ixy, Iyy = calculate_the_second_moment_matrix(horizontal_derivatives, vertical_derivatives)
    
    # Compute corner response
    corner_response = calculate_corner_response(image, Ixx, Ixy, Iyy)
    
    # Threshold the corner response to identify corners
    corners = np.argwhere(corner_response > threshold)
    
    # Apply non-maximum suppression
    suppressed_corners = non_max_suppression(corners, corner_response)
    
    # Create a blank binary image
    binary_image = np.zeros_like(image, dtype=np.uint8)
    # print("sup. cor", suppressed_corners)
    # Mark the suppressed corners on the binary image
    for corner in suppressed_corners:
        y, x = corner  # Swap x and y to match array indexing
        binary_image[y, x] = 255  # Set pixel to white (255) at corner position
    
    return binary_image









def detect_corners(image_path, threshold=0.01):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform corner detection using the Harris corner detection algorithm
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Threshold the corner response to retain only strong corners
    corner_image = np.zeros_like(image)
    corner_image[corners > threshold * corners.max()] = [0, 0, 255]  # Draw detected corners as red dots

    return corner_image

# Example usage:
image_path = 'pillsetc.jpg'
original_image = cv2.imread(image_path)
detected_corners = detect_corners(image_path)
detected_corners = detect_corners_binary(image_path)
cv2.imshow('original image', original_image)
cv2.imshow('Detected Corners', detected_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()



detected_corners = detect_corners(image_path)
cv2.imshow('Detected Corners with python_package', detected_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Call non-maximum suppression to detect corners
# suppressed_corners = non_max_suppression(corners, corners)
# print("Suppressed corners after non-maximum suppression:", suppressed_corners)


