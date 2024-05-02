import cv2
import numpy as np
from scipy.signal import convolve2d

from skimage.transform import hough_line, hough_line_peaks

def gaussian_derivative_kernels(sigma):
    size = 3  # Size of the kernel (3x3)
    center = size // 2  # Center of the kernel

    # Create empty kernels
    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))

    # Calculate the constant factor
    constant = 1 / (np.sqrt(2 * np.pi) * sigma**3)

    # Calculate the Gaussian values for each element in the kernels
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center

            # Compute the value for the horizontal (x) direction
            kernel_x[i, j] = -x * constant * np.exp(-(x**2 + y**2) / (2 * sigma**2))

            # Compute the value for the vertical (y) direction
            kernel_y[i, j] = -y * constant * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernels
    kernel_x /= np.sum(np.abs(kernel_x))
    kernel_y /= np.sum(np.abs(kernel_y))

    return kernel_x, kernel_y

def calculate_gradients(image, sigma=10):
    # Compute Gaussian derivative kernels
    kernel_x, kernel_y = gaussian_derivative_kernels(sigma)

    # Calculate horizontal and vertical derivatives
    horizontal_derivatives = convolve2d(image, kernel_x, mode='same')
    vertical_derivatives = convolve2d(image, kernel_y, mode='same')

    # Calculate magnitude and orientation
    magnitude = np.sqrt(horizontal_derivatives**2 + vertical_derivatives**2)
    orientation = np.arctan2(vertical_derivatives, horizontal_derivatives)

    return magnitude, orientation

def non_maximum_suppression(magnitude, orientation):
    rows, cols = magnitude.shape
    suppressed_magnitude = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = orientation[i, j]
            mag = magnitude[i, j]

            # Determine neighbor indices based on orientation
            if (angle >= 0 and angle < np.pi / 4) or (angle >= 7 * np.pi / 4 and angle < 2 * np.pi):
                neighbor1 = magnitude[i, j + 1]
                neighbor2 = magnitude[i, j - 1]
            elif (angle >= np.pi / 4 and angle < 3 * np.pi / 4):
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]
            elif (angle >= 3 * np.pi / 4 and angle < 5 * np.pi / 4):
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            else:
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            # Perform non-maximum suppression
            if mag >= neighbor1 and mag >= neighbor2:
                suppressed_magnitude[i, j] = mag
            else:
                suppressed_magnitude[i, j] = 0

    # Resize the magnitude image to match the dimensions of the original image
    suppressed_magnitude_resized = cv2.resize(suppressed_magnitude, (cols, rows))
    
    return suppressed_magnitude_resized

def threshold_and_linking(suppressed_magnitude, low_threshold=50, high_threshold=150):
    # Create a binary image for edges
    edges_binary = np.zeros_like(suppressed_magnitude)

    # Apply high threshold to identify strong edges
    strong_edges = suppressed_magnitude >= high_threshold

    # Apply low threshold to identify weak edges
    weak_edges = (suppressed_magnitude >= low_threshold) & (suppressed_magnitude < high_threshold)

    # Perform linking (hysteresis)
    for i in range(1, suppressed_magnitude.shape[0] - 1):
        for j in range(1, suppressed_magnitude.shape[1] - 1):
            if strong_edges[i, j]:
                edges_binary[i, j] = 255
            elif weak_edges[i, j]:
                # Check if any of the 8-connected neighbors are strong edges
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges_binary[i, j] = 255

    return edges_binary



def detect_edge_my_impl(image, image_original):
    # # Calculate gradients
    # magnitude, orientation = calculate_gradients(image)

    # # Perform non-maximum suppression
    # suppressed_magnitude = non_maximum_suppression(magnitude, orientation)

    # # Thresholding and linking
    # edges_binary = threshold_and_linking(suppressed_magnitude, low_threshold=50, high_threshold=150)

    # # Normalize the magnitude for visualization
    # magnitude_normalized = cv2.normalize(suppressed_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # # Convert the magnitude image to 3-channel RGB format
    # magnitude_rgb = cv2.cvtColor(magnitude_normalized, cv2.COLOR_GRAY2RGB)

    # # Overlay the edges on the original image
    # result = cv2.addWeighted(image_original, 0.5, magnitude_rgb, 0.5, 0)
     # Calculate gradients
    magnitude, orientation = calculate_gradients(image)

    # Perform non-maximum suppression
    suppressed_magnitude = non_maximum_suppression(magnitude, orientation)

    # Thresholding
    _, edges_binary = cv2.threshold(suppressed_magnitude, 50, 255, cv2.THRESH_BINARY)

    # Display the result
    cv2.imshow('Binary Edges Image', edges_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  

    # Display the result
    cv2.imshow('Edges Highlighted on Image', edges_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges_binary


def detect_edges(image_path, low_threshold=50, high_threshold=150):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert single-channel edges image to three-channel format for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_rgb


def hough_transform(edges_binary):
    # Define the resolution for rho and theta parameters
    rho_resolution = 1
    theta_resolution = np.pi / 180
    
    # Define the threshold for line detection
    threshold = 100
    
    # Initialize accumulator array
    # rows, cols = edges_binary.shape  # Get the shape of the binary image
    rows, cols = edges_binary[0],  edges_binary[1],  # Get the shape of the binary image
    diagonal_length = np.ceil(np.sqrt(rows ** 2 + cols ** 2))
    num_thetas = int(np.pi / theta_resolution)
    accumulator = np.zeros((int(2 * diagonal_length), num_thetas), dtype=np.uint64)

    # Iterate over each edge pixel
    edge_points = np.argwhere(edges_binary > 0)
    for y, x in edge_points:
        # Iterate over each possible theta value
        for theta_index in range(num_thetas):
            theta = theta_index * theta_resolution
            
            # Calculate rho value
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + diagonal_length
            
            # Increment the accumulator
            accumulator[rho, theta_index] += 1

    # Find lines in the accumulator exceeding the threshold
    lines_image = np.zeros_like(edges_binary)
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] > threshold:
                rho = rho_index - diagonal_length
                theta = theta_index * theta_resolution
                
                # Calculate endpoints of the line
                x0 = int(rho * np.cos(theta))
                y0 = int(rho * np.sin(theta))
                x1 = int(x0 + 1000 * (-np.sin(theta)))
                y1 = int(y0 + 1000 * (np.cos(theta)))
                x2 = int(x0 - 1000 * (-np.sin(theta)))
                y2 = int(y0 - 1000 * (np.cos(theta)))
                
                # Draw the line on the output image
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

    return lines_image

def hough_transform(edges_binary):
    # Convert edges_binary to 8-bit unsigned integer
    edges_binary = np.uint8(edges_binary)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLines(edges_binary, rho=1, theta=np.pi/180, threshold=100)

    # Draw the detected lines on a blank image
    lines_image = np.zeros_like(edges_binary)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = int(a * rho)
            y0 = int(b * rho)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

    # Display the lines image
    cv2.imshow('Hough Lines', lines_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return lines_image

def hough_transform_skimage(edges_binary):
    # Perform Hough transform to detect lines
    h, theta, d = hough_line(edges_binary)

    # Get peaks from Hough transform accumulator
    _, angles, dists = hough_line_peaks(h, theta, d)

    # Create a blank image to draw detected lines
    lines_image = np.zeros_like(edges_binary)

    # Draw detected lines
    for _, angle, dist in zip(range(10), angles, dists):
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)
        x1 = int(x0 + 1000 * (-np.sin(angle)))
        y1 = int(y0 + 1000 * (np.cos(angle)))
        x2 = int(x0 - 1000 * (-np.sin(angle)))
        y2 = int(y0 - 1000 * (np.cos(angle)))
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

    return lines_image


image_path = 'hinges.jpg'
original_image = cv2.imread(image_path)
cv2.imshow('original image', original_image)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_original = cv2.imread(image_path)
edges_binary = detect_edge_my_impl(image, image_original)

detected_edges = detect_edges(image_path)
cv2.imshow('Detected Edges using python package', detected_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform Hough Transform on the identified edges
#hough_transform(edges_binary)
hough_lines_image = hough_transform_skimage(image)

# Display the result
cv2.imshow('Hough Transform Result using python package', hough_lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


