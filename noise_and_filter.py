import numpy as np
import cv2
from skimage.util import random_noise


def median_filter(img, kernel_size):
    d = int((kernel_size - 1) / 2)
    new_img = img.copy()
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            neighbor = []
            for i in range(-d, d + 1):
                if 0 <= m + i < img.shape[0]:
                    for j in range(-d, d + 1):
                        if 0 <= (n + j) < img.shape[1]:
                            neighbor += [img[m + i][n + j]]
            neighbor = np.array(neighbor)
            # 取中位數作為新值
            val = np.median(neighbor)
            new_img[m][n] = val
    return new_img


def neighbor_average(img, kernel_size):
    d = int((kernel_size - 1) / 2)
    total = kernel_size * kernel_size
    new_img = img.copy()
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            neighbor = []
            for i in range(-d, d + 1):
                if 0 <= m + i < img.shape[0]:
                    for j in range(-d, d + 1):
                        if 0 <= (n + j) < img.shape[1]:
                            neighbor += [img[m + i][n + j]]
            neighbor = np.array(neighbor)
            # 取均值作為新值
            val = np.sum(neighbor) / total
            new_img[m][n] = val
    return new_img


def unsharp_mask(img):
    d = 1
    new_img = img.copy()
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            sum = 0
            for i in range(-d, d + 1):
                if m + i >= 0 and m + i < img.shape[0]:
                    for j in range(-d, d + 1):
                        if (n + j) >= 0 and (n + j) < img.shape[1]:
                            sum += -1 * img[m + i][n + j]
            sum += 9 * img[m][n]
            sum /= 9
            new_img[m][n] = sum
    return new_img


def salt_(img, proportion=0.1):
    ind = np.random.randint(0, img.size, int(img.size * proportion))
    x, y = ind % img.shape[0], np.floor(ind / img.shape[0])
    x = x.astype(int)
    y = y.astype(int)
    new_img = img.copy()
    for i in range(len(x)):
        new_img[x[i], y[i]] = 255

    cv2.imshow('s', new_img)
    return new_img


def pepper_(img, proportion=0.1):
    ind = np.random.randint(0, img.size, int(img.size * proportion))
    x, y = ind % img.shape[0], np.floor(ind / img.shape[0])
    x = x.astype(int)
    y = y.astype(int)
    new_img = img.copy()
    for i in range(len(x)):
        new_img[x[i], y[i]] = 0

    cv2.imshow('s', new_img)
    return new_img


if __name__ == "__main__":
    dataSavePath = "./data_after_process/noise/"
    img = cv2.imread("./data/small_pokemon_ball.jpg", 0)
    # gaussian_img = cv2.imread("./data_after_process/noise/gaussian_img.jpg", 0)

    # sp_img = random_noise(img, mode='s&p')
    sp_img = cv2.imread("./data_after_process/erosion_dilation/s&p.jpg", 0)
    sp_img = (255 * sp_img).astype(np.uint8)
    cv2.imwrite(dataSavePath + "sp_img.jpg", sp_img)
    med_img = median_filter(sp_img, 3)
    cv2.imwrite(dataSavePath + "s&p_median filter_3.jpg", med_img)

    # neighbor_average
    avg_img = neighbor_average(sp_img, 3)
    cv2.imwrite(dataSavePath + "s&p_avg filter_3.jpg", avg_img)

    '''cv2.imshow('s&p', sp_img)
    gaussian_img = random_noise(img, mode='gaussian', var=0.05 ** 2)
    gaussian_img = (255 * gaussian_img).astype(np.uint8)
    #cv2.imshow('gaussian', gaussian_img)
    cv2.imwrite(dataSavePath + "gaussian_img.jpg", gaussian_img)'''
    '''#median_fillter
    med_img = median_fillter(gaussian_img, 3)
    cv2.imwrite(dataSavePath + "median filter_3.jpg", med_img)
    med_img=median_fillter(gaussian_img,5)
    cv2.imwrite(dataSavePath + "median filter_5.jpg", med_img)
    med_img = median_fillter(gaussian_img, 7)
    cv2.imwrite(dataSavePath + "median filter_7.jpg", med_img)
    
    #neighbor_average
    avg_img = neighbor_average(gaussian_img, 3)
    cv2.imwrite(dataSavePath + "avg filter_3.jpg", avg_img)
    avg_img = neighbor_average(gaussian_img, 5)
    cv2.imwrite(dataSavePath + "avg filter_5.jpg", avg_img)
    avg_img = neighbor_average(gaussian_img, 7)
    cv2.imwrite(dataSavePath + "avg filter_7.jpg", avg_img)'''
    '''blur_img=cv2.imread("./data/j.png", 0)
    sharp_img=unsharp_mask(blur_img)
    cv2.imwrite(dataSavePath + "unsharp_mask.jpg", sharp_img)'''

    # cv2.waitKey(0)
