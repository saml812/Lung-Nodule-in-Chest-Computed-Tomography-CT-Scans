import numpy as np

# Pixel by pixel
# def contrast_stretching(I, t, Imin, Imax, Lmin=0, Lmax=255):
#     if I <= t:
#         L = (I - Imin) * ((t - Lmin) / (t - Imin)) + Lmin
#     else:
#         L = (I - t + 1) * ((Lmax - t + 1) / (Imax - t + 1)) + t + 1
    
#     return np.clip(L, 0, 255).astype(np.uint8)

# vectorized
def contrast_stretching(img, t, Imin, Imax, Lmin=0, Lmax=255):
    out = np.zeros_like(img, dtype=np.double)
    low_mask = img <= t
    high_mask = img > t

    #[Imin, t] -> [Lmin, t]
    if np.any(low_mask):
        if Imin == t:
            out[low_mask] = Lmin # avoid division by zero
        else:
            out[low_mask] = (img[low_mask] - Imin) * ((t - Lmin) / (t - Imin)) + Lmin

    #(t, Imax] -> (t, Lmax]
    if np.any(high_mask):
        if Imax == t + 1:
            out[high_mask] = Lmax # avoid division by zero
        else:
            out[high_mask] = (img[high_mask] - (t + 1)) * ((Lmax - (t + 1)) / (Imax - (t + 1))) + (t + 1)

    out = np.clip(out, Lmin, Lmax).astype(np.uint8)
    return out