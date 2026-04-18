import preprocess
import enhance
import measure
import numpy as np

def main():
    images = preprocess.load_dataset("output")
    k1, k2 = 8, 8
    c = 1
    alpha = 0.5
    Lmin, Lmax = 0, 255

    for idx, image in enumerate(images):
        Imin = np.min(image)
        Imax = np.max(image)

        best = {
            'eme': (-np.inf, 0),
            'emee': (-np.inf, 0),
            'ame': (-np.inf, 0),
            'eme_log': (-np.inf, 0),
            'visibility': (-np.inf, 0),
            'amee': (-np.inf, 0),
        }

        for t in range(0, 256):
            if not (Imin < t < Imax):
                continue

            new_image = np.zeros_like(image, dtype=np.uint8)
            rows, cols = image.shape
            for i in range(rows):
                for j in range(cols):
                    I = image[i, j]
                    new_image[i, j] = enhance.contrast_stretching(I, t, Imin, Imax, Lmin, Lmax)

            eme_val = measure.eme(new_image, k1, k2, c)
            emee_val = measure.emee(new_image, k1, k2, c, alpha)
            ame_val = measure.ame(new_image, k1, k2, c)
            eme_log_val = measure.eme_log(new_image, k1, k2, c)
            visibility_val = measure.visibility(new_image, k1, k2, c)
            amee_val = measure.amee(new_image, k1, k2, c, alpha)

            if eme_val > best['eme'][0]:
                best['eme'] = (eme_val, t)
            if emee_val > best['emee'][0]:
                best['emee'] = (emee_val, t)
            if ame_val > best['ame'][0]:
                best['ame'] = (ame_val, t)
            if eme_log_val > best['eme_log'][0]:
                best['eme_log'] = (eme_log_val, t)
            if visibility_val > best['visibility'][0]:
                best['visibility'] = (visibility_val, t)
            if amee_val > best['amee'][0]:
                best['amee'] = (amee_val, t)

        print(f"Image {idx+1} best t values:")
        for metric, (val, t_opt) in best.items():
            print(f"  {metric}: t = {t_opt} (value = {val:.4f})")

if __name__ == "__main__":
    main()