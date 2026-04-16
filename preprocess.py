import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_dataset(path):
    # Read each image
    # Convert each image to grayscale
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)

    return images

def save_images(images, directory):
    for idx, img in enumerate(images):
        path = os.path.join(directory, f"image_{idx}.png")
        cv2.imwrite(path, img)

def median_filter(img, size):
    result = cv2.medianBlur(img, size)
    return result

def gaussian_filter(img, size, sigma):
    result = cv2.GaussianBlur(img, (size, size), sigmaX=sigma)
    return result

def main():
    # Load dataset
    images = load_dataset("dataset")

    # Apply Median filter
    images_median = []
    for image in images:
        images_median.append(median_filter(image, 3))

    # Apply Gaussian filter
    preprocessed = []
    for image in images_median:
        preprocessed.append(gaussian_filter(image, 3, 1))

    # Save preprocessed images
    save_images(preprocessed, "output")
    
if __name__ == "__main__":
    main()