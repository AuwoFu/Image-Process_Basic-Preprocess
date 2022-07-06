import numpy as np
import cv2
from skimage.util import random_noise


def dilate_(img, kernel, iterations=1):
    my_img = img.copy()
    col = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    row = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    for time in range(iterations):
        masked_img = np.pad(my_img, (1, 1), "constant")
        for i in range(1, masked_img.shape[0] - 1):
            for j in range(1, masked_img.shape[1] - 1):
                window = masked_img[col + i, row + j]
                cp_mask = np.bitwise_and(window, kernel)
                if (cp_mask == 255).any():
                    my_img[i - 1, j - 1] = 255

    return my_img


def erosion_(img, kernel, iterations=1):
    my_img = img.copy()
    col = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    row = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    for time in range(iterations):
        masked_img = np.pad(my_img, (1, 1), "constant")
        for i in range(1, masked_img.shape[0] - 1):
            for j in range(1, masked_img.shape[1] - 1):
                window = masked_img[col + i, row + j]
                print(window)
                print(kernel)
                cp_mask = np.bitwise_and(window, kernel)
                if (cp_mask == kernel).all():
                    my_img[i - 1, j - 1] = 255
                else:
                    my_img[i - 1, j - 1] = 0

    return my_img


def open_(img, kernel, iterations=1):
    my_img = erosion_(img, kernel, iterations)
    my_img = dilate_(my_img, kernel, iterations)
    return my_img


def close_(img, kernel, iterations=1):
    my_img = dilate_(img, kernel, iterations)
    my_img = erosion_(my_img, kernel, iterations)
    return my_img


def convert_to_binary(img, thres=127):
    my_img = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] < thres:
                my_img[i][j] = 0
            else:
                my_img[i][j] = 255
    return my_img

# get foreground noise(which darker than original)
def morphology_blackhat(img, kernel, iterations):
    my_img = close_(img, kernel, iterations)
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            my_img[i][j] = int(my_img[i][j]) - int(img[i][j])
    return my_img

# get background noise(which brighter than original)
def morphology_tophat(img, kernel, iterations):
    my_img = open_(img, kernel, iterations)
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            my_img[i][j] = int(my_img[i][j])-int(img[i][j])
    return my_img


def morphology_gradient(img, kernel, iterations):
    d_img=dilate_(img,kernel,iterations)
    e_img=erosion_(img,kernel,iterations)
    my_img = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    for i in range(my_img.shape[0]):
        for j in range(my_img.shape[1]):
            my_img[i][j] = int(d_img[i][j])-int(e_img[i][j])
    return my_img

def change2binary(img):
    new_img=img.copy()
    for i in range(img.shape[0]):
        g=img[i]
        ind = np.zeros(img.shape[1])
        ind[g<127]=0
        ind[g>=127]=1
        new_img[i][ind==0]=0
        new_img[i][ind == 1] = 255
    return new_img

def salt_(img,proportion=0.1):
    ind=np.random.randint(0,img.size,int(img.size*proportion))
    x,y=ind%img.shape[0],np.floor(ind/img.shape[0])
    x=x.astype(int)
    y=y.astype(int)

    new_img=img.copy()
    for i in range(len(x)):
        new_img[x[i],y[i]]=255

    cv2.imshow('s',new_img)
    return new_img





if __name__ == "__main__":
    dataSavePath="./data_after_process/erosion_dilation/"
    col = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    row = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # erosion & dilation
    kernel = np.array([
        [0, 255, 0],
        [255, 255, 255],
        [0, 255, 0]
    ], dtype=np.uint8)

    img = cv2.imread("./data/j.png", 0)
    '''dilation_output = dilate_(img, kernel, iterations=3)
    erosion_output = erosion_(img, kernel, iterations=3)
    opening_output = open_(img, kernel, iterations=2)
    closing_output = close_(img, kernel, iterations=2)
    cv2.imwrite(dataSavePath+"dilation.jpg", dilation_output)
    cv2.imwrite(dataSavePath+"erosion.jpg", erosion_output)
    cv2.imwrite(dataSavePath+"opening.jpg", opening_output)
    cv2.imwrite(dataSavePath+"closing.jpg", closing_output)

    blackhat_output = morphology_blackhat(img, kernel, iterations=3)
    tophat_output = morphology_tophat(img, kernel, iterations=3)
    gradient_output = morphology_gradient(img, kernel, iterations=1)
    cv2.imwrite(dataSavePath+"morphology_blackhat.jpg", blackhat_output)
    cv2.imwrite(dataSavePath+"morphology_tophat.jpg", tophat_output)
    cv2.imwrite(dataSavePath+"morphology_gradient.jpg", gradient_output)
    '''
    s_img = salt_(img)
    #cv2.imshow('s&p', sp_img)
    cv2.imwrite(dataSavePath + "s.jpg", s_img)
    dilation_output = dilate_(s_img, kernel, iterations=1)
    erosion_output = erosion_(s_img, kernel, iterations=1)
    opening_output = open_(s_img, kernel, iterations=1)
    closing_output = close_(s_img, kernel, iterations=1)
    cv2.imwrite(dataSavePath + "s_dilation.jpg", dilation_output)
    cv2.imwrite(dataSavePath + "s_erosion.jpg", erosion_output)
    cv2.imwrite(dataSavePath + "s_opening.jpg", opening_output)
    cv2.imwrite(dataSavePath + "s_closing.jpg", closing_output)

    blackhat_output = morphology_blackhat(s_img, kernel, iterations=1)
    tophat_output = morphology_tophat(s_img, kernel, iterations=1)
    gradient_output = morphology_gradient(s_img, kernel, iterations=1)
    cv2.imwrite(dataSavePath + "s_morphology_blackhat.jpg", blackhat_output)
    cv2.imwrite(dataSavePath + "s_morphology_tophat.jpg", tophat_output)
    cv2.imwrite(dataSavePath + "s_morphology_gradient.jpg", gradient_output)

    cv2.waitKey(0)
