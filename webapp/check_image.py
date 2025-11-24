import sys
import cv2
import numpy as np
p = 'webapp/last_annotated.jpg'
img = cv2.imread(p)
if img is None:
    print('MISSING')
    sys.exit(1)
print('path=', p)
print('shape=', img.shape)
print('dtype=', img.dtype)
print('mean=', float(np.mean(img)))
h,w = img.shape[:2]
cy = img[h//4:3*h//4, w//4:3*w//4]
print('center mean=', float(np.mean(cy)))
