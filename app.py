import argparse
import numpy as np

import cv2
import matplotlib.pyplot as plt

img = None
selection = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    global selection
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        selection.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
        selection.append((x, y))


def unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

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


def run(file_path):
    global img
    img = cv2.imread(file_path)
    img = cv2.resize(img, (800, 600))
    h, w = img.shape[0], img.shape[1]

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    src = np.float32(selection)
    dst = np.float32([(w, 0),  # pd
                      (0, 0),
                      (w, h),
                      (0, h)])

    unwarp(img, src, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='./kostka.jpg')

    args = parser.parse_args()

    run(args.file_path)
