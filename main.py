import preprocess
import enhance
import numpy as np

def main():
    images = preprocess.load_dataset("output")
    
    Lmin = 0
    Lmax = 255

    for image in images:
        Imin = np.min(image)
        Imax = np.max(image)
        for t in range(0,255):
            new_image = np.zeros(image)
            for i in range(len(image)):
                for j in range(len(image[0])):
                    I = image[i,j]
                    new_image[i,j] = enhance.contrast_stretching(I, t, Imin, Imax, Lmin, Lmax)
            # Apply the measure functions
            


if __name__ == "__main__":
    main()