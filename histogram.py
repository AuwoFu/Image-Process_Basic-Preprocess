import numpy as np
import cv2
from math import pi
import matplotlib.pyplot as plt


def histogram_equalization(img):
    my_img = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    gl_count = np.zeros(256, np.int32)

    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            gl_count[int(img[i][j])] += 1
    plt.plot(gl_count, c='b')

    '''for i in range(256):
        if gl_count[i] != 0:
            print('range min: ', i)
            break
    for i in range(255, -1, -1):
        if gl_count[i] != 0:
            print('range max: ', i)
            break'''

    # 像素出現機率
    pdf = gl_count / my_img.size
    cdf = np.zeros(256)
    t = 0
    for i in range(256):
        t += pdf[i]
        cdf[i] = t

    # 根據cdf 分配新灰階值
    gl_count = np.zeros(256, np.int32)
    new_gl_value = np.around(cdf * 255).astype('uint8')
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            my_img[i][j] = new_gl_value[int(img[i][j])]
            gl_count[int(my_img[i][j])] += 1
    plt.plot(gl_count, c='r')
    # plt.show()
    return my_img


# Contrast Limited Adaptive Histogram Equalization
def CLAHE(img, threshold):
    my_img = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    gl_count = np.zeros(256)
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            gl_count[int(img[i][j])] += 1
    '''for i in range(256):
        if gl_count[i] != 0:
            print('range min: ', i)
            break
    for i in range(255, -1, -1):
        if gl_count[i] != 0:
            print('range max: ', i)
            break'''

    # 像素出現機率
    pdf = gl_count / my_img.size
    plt.plot(pdf.copy(), c='blue')
    print(pdf)
    redistrubute = 0
    for i in range(256):
        if pdf[i] > threshold:
            redistrubute += (pdf[i] - threshold)
            pdf[i] -= threshold
    redistrubute /= 256
    pdf += redistrubute
    print(pdf)
    plt.plot(pdf, c='g')
    plt.show()

    cdf = np.zeros(256)
    t = 0
    for i in range(256):
        t += pdf[i]
        cdf[i] = t

    # 根據cdf 分配新灰階值
    new_gl_value = np.around(cdf * 255).astype('uint8')
    plt.figure()
    plt.plot(gl_count, c='b')
    gl_count = np.zeros(256)
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            my_img[i][j] = int(new_gl_value[int(img[i][j])])
            gl_count[my_img[i][j]] += 1
    plt.plot(gl_count, c='r')
    print("CLAHE")
    # plt.show()
    return my_img


def AHE(img, blocksize):
    # w,h of block;
    w, h = int(img.shape[0] / blocksize[0]), int(img.shape[1] / blocksize[1])
    new_img = img.copy()

    # 將原圖切割成小塊, 進行HE
    for i in range(1, blocksize[0] + 1):
        for j in range(1, blocksize[1] + 1):
            subimg = img[w * (i - 1):w * i, h * (j - 1):h * j]
            subimg = histogram_equalization(subimg)
            new_img[w * (i - 1):w * i, h * (j - 1):h * j] = subimg.copy()
    plt.figure()
    gl_count = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gl_count[img[i][j]] += 1
    plt.plot(gl_count, c='b')

    gl_count = np.zeros(256)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            gl_count[new_img[i][j]] += 1
    plt.plot(gl_count, c='r')
    print("AHE")
    plt.show()
    return new_img


def HE_in_RGB(img):
    '''
    will change the Proportion of three basic color
    :param img: RGB img
    :return: RGB img
    '''
    my_img = np.zeros([img.shape[0], img.shape[1], 3], np.int64)
    B, G, R = cv2.split(img)
    my_img[:, :, 0] = histogram_equalization(B)
    my_img[:, :, 1] = histogram_equalization(G)
    my_img[:, :, 2] = histogram_equalization(R)
    return my_img


def HE_in_HSI(img):
    my_img = img.copy()
    I = my_img[:, :, 2] * 255
    new_I = histogram_equalization(I)
    my_img[:, :, 2] = new_I / 255
    return my_img


def AHE_in_RGB(img, blocksize):
    my_img = np.zeros([img.shape[0], img.shape[1], 3], np.int64)
    my_img[:, :, 0] = AHE(img[:, :, 0], blocksize)
    my_img[:, :, 1] = AHE((img[:, :, 1]), blocksize)
    my_img[:, :, 2] = AHE(img[:, :, 2], blocksize)
    return my_img


def AHE_in_HSI(img, blocksize):
    my_img = img.copy()
    I = my_img[:, :, 2] * 255
    new_I = AHE(I, blocksize)
    my_img[:, :, 2] = new_I / 255
    return my_img


def CLAHE_in_RGB(img, threshold):
    '''
    will change the Proportion of three basic color
    :param img: RGB img
    :return: RGB img
    '''
    my_img = np.zeros([img.shape[0], img.shape[1], 3], np.int64)
    my_img[:, :, 0] = CLAHE(img[:, :, 0], threshold)
    my_img[:, :, 1] = CLAHE((img[:, :, 1]), threshold)
    my_img[:, :, 2] = CLAHE(img[:, :, 2], threshold)
    return my_img


def CLAHE_in_HSI(img, threshold):
    my_img = img.copy()
    I = my_img[:, :, 2] * 255
    new_I = CLAHE(I, threshold)
    my_img[:, :, 2] = new_I / 255
    return my_img


def RGB2HSI(img):
    hsi_img = np.zeros([img.shape[0], img.shape[1], 3])
    B, G, R = cv2.split(img)
    [B, G, R] = [i / 255 for i in ([B, G, R])]

    H = np.zeros([img.shape[0], img.shape[1]])
    S = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        b, g, r = B[i], G[i], R[i]
        temp = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
        theta = np.arccos(0.5 * (r - b + r - g) / temp)
        # H
        h = np.zeros(img.shape[1])
        h[g >= b] = theta[g >= b]
        h[g < b] = 2 * np.pi - theta[g < b]
        h[temp == 0] = 0
        H[i] = h / (2 * np.pi)

        # S
        min = []
        for j in range(img.shape[1]):
            min.append(np.min([r[j], g[j], b[j]]))
        min = np.array(min)
        S[i] = 1 - 3 * min / (r + g + b)
        S[i][(r + g + b) == 0] = 0

    hsi_img[:, :, 0] = H
    hsi_img[:, :, 1] = S
    # I
    hsi_img[:, :, 2] = (R + G + B) / 3
    return hsi_img


def HSI2BGR(img):
    rgb_img = np.zeros([img.shape[0], img.shape[1], 3])
    H, S, I = cv2.split(img.copy())
    # 調整成[0,1]
    # [H, S, I] = [i / 255 for i in ([H, S, I])]
    R, G, B = cv2.split(rgb_img)
    for m in range(img.shape[0]):
        h, s, i = H[m], S[m], I[m]
        h = h * (2 * np.pi)
        ind = np.zeros(img.shape[1])
        for n in range(img.shape[1]):
            if h[n] >= 4 * pi / 3:
                h[n] -= 4 * pi / 3
                ind[n] = 3
            elif h[n] >= 2 * pi / 3:
                h[n] -= 2 * pi / 3
                ind[n] = 2
            else:
                ind[n] = 1
        a = i * (1 - s)
        b = i * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h))
        c = 3 * i - a - b
        # type == 1:
        B[m][ind == 1] = a[ind == 1]
        R[m][ind == 1] = b[ind == 1]
        G[m][ind == 1] = c[ind == 1]
        # type == 2:
        B[m][ind == 2] = c[ind == 2]
        R[m][ind == 2] = a[ind == 2]
        G[m][ind == 2] = b[ind == 2]
        # type == 3:
        B[m][ind == 3] = b[ind == 3]
        R[m][ind == 3] = c[ind == 3]
        G[m][ind == 3] = a[ind == 3]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img


if __name__ == "__main__":
    dataSavePath = "./data_after_process/contrast/"
    img = cv2.imread("./data/street.jpg")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dataSavePath + "original gray level.jpg", gray_img)
    # HE
    '''
    gray_output = histogram_equalization(gray_img)
    cv2.imwrite(dataSavePath + "HE in gray level.jpg", gray_output)
    cv2.waitKey(0)
    HE_in_RGB_output = HE_in_RGB(img)
    cv2.imwrite(dataSavePath + "HE in RGB.jpg", HE_in_RGB_output)

    HSI_img = RGB2HSI(img)
    HSI_output = HE_in_HSI(HSI_img)
    HE_output = HSI2BGR(HSI_output)
    
    cv2.imwrite(dataSavePath + "HE in HSI.jpg", HE_output)
    # AHE
    print('start AHE')
    gray_output = AHE(gray_img,[5,4])
    cv2.imwrite(dataSavePath + "AHE in gray level.jpg", gray_output)
    HE_in_RGB_output = AHE_in_RGB(img,[5,4])
    cv2.imwrite(dataSavePath + "AHE in RGB.jpg", HE_in_RGB_output)

    HSI_img = RGB2HSI(img)
    HSI_output = AHE_in_HSI(HSI_img,[5,4])
    HE_output = HSI2BGR(HSI_output)
    cv2.imwrite(dataSavePath + "AHE in HSI.jpg", HE_output)

    # CLAHE
    threshold = 0.01
    gray_output = CLAHE(gray_img, threshold)
    cv2.imwrite(dataSavePath + "CLAHE in gray level.jpg", gray_output)

    HE_in_RGB_output = CLAHE_in_RGB(img, threshold)
    cv2.imwrite(dataSavePath + "CLAHE in RGB.jpg", HE_in_RGB_output)

    HSI_img = RGB2HSI(img)
    HSI_output = CLAHE_in_HSI(HSI_img, threshold)
    HE_output = HSI2BGR(HSI_output)
    cv2.imwrite(dataSavePath + "CLAHE in HSI.jpg", HE_output)'''
    cv2.waitKey(0)
