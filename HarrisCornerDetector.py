import numpy as np
import cv2
import matplotlib.pyplot as plt

def harris_corner_detector(image, k=0.4, window_size=5, threshold=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=window_size)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=window_size)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    corner_response = np.zeros_like(gray)
    for i in range(height):
        for j in range(width):
            Sxx = np.sum(Ixx[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1])
            Syy = np.sum(Iyy[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1])
            Sxy = np.sum(Ixy[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1])
            det = Sxx * Syy - Sxy**2
            trace = Sxx + Syy
            corner_response[i, j] = det - k * trace**2
    corner_response = np.where(corner_response > threshold * corner_response.max(), corner_response, 0)
    for i in range(height):
        for j in range(width):
            if corner_response[i, j] != 0:
                cv2.circle(image, (j, i), 5, (0, 0, 255), -1)
    return image



k_values = [0.4]
window_size_values = [3,5,7]
threshold_values = [0.01,0.1,0.2,0.4,0.5,0.7,0.8,0.9,0.99,0.999,0.9999,0.99999]

for k in k_values:
    for w in window_size_values:
        for t in threshold_values:
            image = cv2.imread('1.png')
            result = harris_corner_detector(image)

            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"C:\\Danial\\Projects\\Danial\\Panorama\\harris_results\\harris_k{k}_window{w}_threshold{t}.jpg", result)
            # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # plt.title('Harris Corner Detection')
            # plt.show()
