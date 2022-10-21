import cv2
import numpy as np
# from skimage import transform as trans


def text_align(img, quad, margin_rate=0.02):
            
    image_size=(img.shape[0], img.shape[1])
    src = quad.astype(np.float32)

    text_w = max(np.linalg.norm(quad[0]-quad[3]), np.linalg.norm(quad[1]-quad[2]))
    text_h = max(np.linalg.norm(quad[0]-quad[1]), np.linalg.norm(quad[2]-quad[3]))
    text_size = [int(text_h)+1, int(text_w)+1]

    dst = np.array([
        [text_size[0]*margin_rate,      text_size[1]*margin_rate],
        [text_size[0]*(1-margin_rate),  text_size[1]*margin_rate],
        [text_size[0]*(1-margin_rate),  text_size[1]*(1-margin_rate)],
        [text_size[0]*margin_rate, text_size[1]*(1-margin_rate)]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, image_size)
    warped = warped[:text_size[1],:text_size[0],:]

    return warped