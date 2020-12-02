import cv2
import matplotlib.pyplot as plt
import numpy as np


def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M

path = r'kostka.jpg'
im = cv2.imread(path)
#w, h = im.shape[:2]
w, h = im.shape[0], im.shape[1]

src = np.float32([(57,  917),#ld
                (1991,    917),#pD
                  (500,  435),#lg
                  (1405, 429)])#pg

dst = np.float32([(2048, 0),#pd
                  (0, 0),
                  (2048, 922),
                  (0, 922)])

unwarp(im, src, dst, True)
#cv2.imwrite('korekcja1.jpg', im)
#cv2.imshow("so", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
