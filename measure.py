import numpy as np
import preprocess

def eme(img, k1, k2, c):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    eme = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            eme += I_max / (I_min + c)

    eme = eme / total_blocks
    return eme

def emee(img, k1, k2, c, alpha):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    emee = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if (I_max - I_min == 0):
                continue

            emee += alpha * np.log(I_max / (I_min + c))

    emee = emee / total_blocks
    return emee

def ame(img, k1, k2, c):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    ame = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)
            
            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if (I_max - I_min == 0):
                continue

            ame += np.log((I_max - I_min) / (I_max + I_min + c))

    ame = ame / total_blocks
    return ame

def eme_log(img, k1, k2, c):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    eme_log = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if (I_max - I_min == 0):
                continue

            eme_log += 20 * np.log(I_max / (I_min + c))

    eme_log = eme_log / total_blocks
    return eme_log

def visibility(img, k1, k2, c):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    visibility = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            visibility += (I_max - I_min) / (I_max + I_min + c)

    return visibility

def amee(img, k1, k2, c, alpha):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    amee = 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if (I_max - I_min == 0):
                continue

            amee += np.log(((I_max - I_min) / (I_max + I_min + c)) ** alpha)

    amee = amee / total_blocks
    return amee

def measure_all(img, k1, k2, c, alpha):
    img = img.astype(np.double)

    rows, cols = img.shape
    block_rows = np.floor(rows / k1)
    block_cols = np.floor(cols / k2)

    total_blocks = k1 * k2
    eme, emee, ame, eme_log, visibility, amee = 0, 0, 0, 0, 0, 0

    for k in range(1, k1):
        for l in range(1, k2):
            row_start = int((k - 1) * block_rows)
            row_end   = int(k * block_rows)
            col_start = int((l - 1) * block_cols)
            col_end   = int(l * block_cols)
            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if (I_max - I_min == 0):
                continue

            eme += I_max / (I_min + c)
            emee += alpha * np.log(I_max / (I_min + c))
            ame += np.log((I_max - I_min) / (I_max + I_min + c))
            eme_log += 20 * np.log(I_max / (I_min + c))
            visibility += (I_max - I_min) / (I_max + I_min + c)
            amee += np.log(((I_max - I_min) / (I_max + I_min + c)) ** alpha)

    return (eme/total_blocks, emee/total_blocks, ame/total_blocks, eme_log/total_blocks, visibility, amee/total_blocks)

if __name__ == "__main__":
    images = preprocess.load_dataset("test")

    test_img = images[0]
    print(f"Shape: {test_img.shape}")
    print(f"Data type: {test_img.dtype}")
    print(f"Min: {np.min(test_img)}, Max: {np.max(test_img)}")

    k1, k2 = 20, 20
    c = 1
    alpha = 1

    print("EME      :", eme(test_img, k1, k2, c))
    print("EMEE     :", emee(test_img, k1, k2, c, alpha))
    print("AME      :", ame(test_img, k1, k2, c))
    print("EME (log):", eme_log(test_img, k1, k2, c))
    print("Visibility:", visibility(test_img, k1, k2, c))
    print("AMEE     :", amee(test_img, k1, k2, c, alpha))
