import matplotlib.pyplot as plt
import cv2
import json
import os
import math
import numpy as np

def display_solution(solution):
    # solution = solution.reshape(1,81).tolist()[0]
    plot_dict = dict()
    for i in range(1,10):
        display_image = cv2.imread("data/display/{}.jpg".format(str(i)))
        display_image = cv2.resize(display_image,(90,90))
        plot_dict[str(i)] = display_image
    map = np.zeros(shape=(9*90,9*90,3),dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            number = int(solution[i][j])
            map[90*i:90*(i+1),90*j:(j+1)*90] = plot_dict[str(number)]

    cv2.imshow('solution',map)
    cv2.waitKey(0)


def load_image(image_path,gray_scale=False):
    image = cv2.imread(image_path)
    if gray_scale:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    image = image[..., ::-1]
    return image


def plot_image(image,name,binary=False):
    if not binary:
        plt.imshow(image)
    else:
        plt.imshow(image,cmap='binary')
    plt.title(name)
    plt.show()

def plot_images(list_image,name):
    """
    display many image from the input list image
    @param list_image: list of the image you want to display
    @param name: the title of the plot
    @return: None
    """
    n_images = 9
    fig,axes = plt.subplots(n_images,n_images)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i,ax in enumerate(axes.flat):

        if i >=len(list_image):
            break

        ax.imshow(list_image[i])

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title(name)

    plt.show()

def parse_json(json_path):
    with open(json_path,'r') as f:
        config = json.load(f)
        return config
