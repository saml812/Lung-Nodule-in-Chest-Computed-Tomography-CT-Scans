import preprocess
import enhance
import measure
import numpy as np

def contrast_stretch_vectorized(img, t, Imin, Imax, Lmin, Lmax):
    """Vectorized piecewise linear contrast stretching with safe division."""
    out = np.empty_like(img, dtype=np.float32)
    mask_low = img <= t
    mask_high = img > t

    # Low part: map [Imin, t] -> [Lmin, t]
    if np.any(mask_low):
        if Imin == t:
            # Entire low range collapses to a single output value
            out[mask_low] = Lmin
        else:
            out[mask_low] = Lmin + (t - Lmin) * (img[mask_low] - Imin) / (t - Imin)

    # High part: map (t, Imax] -> (t, Lmax]
    if np.any(mask_high):
        if Imax == t + 1:
            # Entire high range collapses to a single output value
            out[mask_high] = Lmax
        else:
            out[mask_high] = t + 1 + (Lmax - (t + 1)) * (img[mask_high] - (t + 1)) / (Imax - (t + 1))

    # Clip and convert back to original dtype (e.g., uint8)
    out = np.clip(out, Lmin, Lmax).astype(img.dtype)
    return out

def main():
    images = preprocess.load_dataset("output")
    k1, k2 = 20, 20
    c = 1
    alpha = 1
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
            
            stretched = contrast_stretch_vectorized(image, t, Imin, Imax, Lmin, Lmax)

            eme_val, emee_val, ame_val, eme_log_val, visibility_val, amee_val = measure.measure_all(stretched, k1, k2, c, alpha)

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