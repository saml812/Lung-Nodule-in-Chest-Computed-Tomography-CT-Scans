import numpy as np

def eme(img, k1, k2, c):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    total_blocks = k1 * k2
    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += I_max / (I_min + c)

    eme = sum / total_blocks
    return eme

def emee(img, k1, k2, c, alpha):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    total_blocks = k1 * k2
    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += alpha * np.log(I_max / (I_min + c))

    emee = sum / total_blocks
    return emee

def ame(img, k1, k2, c):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    total_blocks = k1 * k2
    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += np.log((I_max - I_min) / (I_max + I_min + c))

    ame = sum / total_blocks
    return ame

def eme_log(img, k1, k2, c):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    total_blocks = k1 * k2
    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += 20 * np.log(I_max / (I_min + c))

    eme_log = sum / total_blocks
    return eme_log

def visibility(img, k1, k2, c):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += (I_max - I_min) / (I_max + I_min + c)

    visibility = sum
    return visibility

def amee(img, k1, k2, c, alpha):
    img = img.astype(np.float64)

    rows, cols = img.shape
    block_rows = rows // k1
    block_cols = cols // k2

    total_blocks = k1 * k2
    sum = 0.0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            sum += np.log(((I_max - I_min) / (I_max + I_min + c)) ** alpha)

    amee = sum / total_blocks
    return amee

