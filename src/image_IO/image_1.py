''' image_1.py. I/O routines for 1-component (grayscale) images.
'''

import numpy as np
import cv2 as cv
import colored
import os
import subprocess
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s() %(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(levelname)s probando %(funcName)s()] %(message)s")
##logger.setLevel(logging.CRITICAL)
##logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

_compression_level = 9 # 0=min, 9=max

def read(fn): # [row, column, component]
    #if __debug__:
        #print(colored.fore.GREEN + f"image_1.read: {fn}", end=' ', flush=True)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    logger.debug(f"{fn} {img.shape} {img.dtype} len={os.path.getsize(fn)} max={img.max()} min={img.min()}")
    return img

def debug_write(img:np.ndarray, fn:str):
    cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, _compression_level])
    len_output = os.path.getsize(fn)
    logger.info(f"image_1.write: {fn} {img.shape} {img.dtype} len={len_output} max={img.max()} min={img.min()}")
    return len_output

def write(img, fn):
    if np.all(img == img[0,0]):
        logger.warning(f"Constant image equal to {img[0,0]}!")
        return 0
    cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, _compression_level])
    if __debug__:
        len_output = os.path.getsize(fn)
        logger.info(f"Before optipng: {len_output} bytes")
    command = f"optipng {fn}"
    logger.debug(command)
    #subprocess.run(["bash", "-c", command], shell=True)
    #subprocess.run([command], shell=True, capture_output=True)
    subprocess.run([command], shell=True)
    len_output = os.path.getsize(fn)
    #if __debug__:
    #    print(colored.fore.GREEN + f"image_1.write: {fn}", img.shape, img.dtype, len_output, img.max(), img.min(), colored.style.RESET)
    logger.info(f"image_1.write: {fn} {img.shape} {img.dtype} len={len_output} max={img.max()} min={img.min()}")
    return len_output

def normalize(img):
    _max = np.max(img)
    _min = np.min(img)
    max_min = _max - _min
    normalized_img = (img - _min) / max_min
    return normalized_img

def get_image_shape(prefix):
    img = read(prefix, 0)
    return img.shape

def print_stats(msg, _max, _min, _avg):
    logger.info(f"{msg} max={_max} min={_min} avg={_avg}")

def show(image, title='', size=(10, 10), fontsize=20):
    plt.figure(figsize=size)
    plt.title(title, fontsize=fontsize)
    plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
    _max, _min, _avg = np.max(image), np.min(image), np.average(image)
    print_stats(title, _max, _min, _avg)

def show_normalized(image, title='', size=(10, 10), fontsize=20):
    plt.figure(figsize=size)
    #plt.imshow(cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB))
    #plt.title(f"{title}\nmax={_max}\nmin={_min}\navg={_avg}", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    _image = normalize(image)
    plt.imshow(255*_image, vmax=255, vmin=0, cmap='gray')
    _max, _min, _avg = np.max(image), np.min(image), np.average(image)
    print_stats(title, _max, _min, _avg)

