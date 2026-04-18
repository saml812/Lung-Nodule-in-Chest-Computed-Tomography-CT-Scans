import numpy as np

def contrast_stretching(I, t, Imin, Imax, Lmin=0, Lmax=255):
    if I <= t:
        L = (I - Imin) * ((t - Lmin) / (t - Imin)) + Lmin
    else:
        L = (I - t + 1) * ((Lmax - t + 1) / (Imax - t + 1)) + t + 1
    
    return np.clip(L, 0, 255).astype(np.uint8)