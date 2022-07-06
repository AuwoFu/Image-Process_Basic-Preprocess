import numpy as np
import cv2


def doing_mask(img, lower, upper):
    [hl, sl, vl] = lower
    [hu, su, vu] = upper
    mask = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            [h, s, v] = img[i][j]
            if hl <= h <= hu and sl <= s <= su and vl <= v <= vu:
                mask[i][j] = 255
    return mask


def filter(img, mask):
    new_img = img.copy()
    new_img[mask == 0] = [0, 0, 0]
    return new_img


def getColorImg(img, color, lower, upper):
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    # res = cv2.bitwise_and(img, img, mask=mask)
    mask = doing_mask(hsv, lower, upper)
    cv2.imshow(color + ' mask', mask)
    cv2.imwrite(dataSavePath + color + " mask.jpg", mask)
    res = filter(img, mask)
    cv2.imshow(color + ' Input', img)
    cv2.imshow(color + ' Result', res)
    cv2.imwrite(dataSavePath + color + ".jpg", res)


if __name__ == "__main__":
    dataSavePath = "./data_after_process/color/"
    img = cv2.imread("./data/fruit.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    getColorImg(img, "green", lower_green, upper_green)

    # orange
    lower = np.array([11, 43, 46])
    upper = np.array([25, 255, 255])
    getColorImg(img, "orange", lower, upper)

    # yellow
    lower = np.array([26, 43, 46])
    upper = np.array([34, 255, 255])
    getColorImg(img, "yellow", lower, upper)

    # red
    lower = np.array([156, 43, 46])
    upper = np.array([180, 255, 255])
    getColorImg(img, "red", lower, upper)

    cv2.waitKey(0)
