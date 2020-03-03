import cv2
from utils.utils import *
import numpy as np
import math

class ImageProcessor:

    def __init__(self,config):
        """
        Constructor of the image processor class
        @param config: the params config of the image processor
        """
        self.config = config

    def convert_to_grayscale(self,rgb_image):
        """
        convert rgb image to grayscale image
        @param rgb_image: the rgb image with format (h,w,c)
        @return: the grayscale of the input image
        """
        grayscale_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
        return grayscale_image

    def remove_noise(self, grayscale_image):
        """
        Remove noise in the grayscale image by applying gaussian blur
        @param grayscale_image: the grayscale image with format (h,w)
        @return: the removed noise image
        """
        k_size = self.config['k_size']
        removed_noise_image = cv2.GaussianBlur(src=grayscale_image,
                                               ksize=(k_size,k_size),
                                               sigmaX=self.config['sigma_X'],
                                               sigmaY=self.config['sigma_Y'])

        return removed_noise_image

    def convert_to_binary(self, grayscale_image):
        """
        convert the grayscale image to the binary image by applying adaptive threshold
        @param grayscale_image: the grayscale image with format (h,w)
        @return: the binary image with format(h,w)
        """
        binary_image = cv2.adaptiveThreshold(grayscale_image,
                                             maxValue=self.config['max_val'],
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=self.config['block_size'],
                                             C=self.config['C'])
        return binary_image

    def apply_morphological_op(self, binary_image, op='dilate'):
        """
        Apply the morphological operator on the binary image
        @param binary_image: the binary image with format  (h,w)
        @param op: the operator you want to apply (dilate,erode,closing,opening)
        @return: the binary image after applying morphological operator
        """
        result = None
        if op == 'dilate':
            k_size = self.config['dilate_k_size']
            iteration = self.config['dilate_iteration']
            kernel = np.ones(shape=(k_size,k_size),dtype=np.uint8)
            result = cv2.dilate(binary_image,kernel=kernel,iterations=iteration)
        elif op =='erode':
            k_size = self.config['erode_k_size']
            iteration = self.config['erode_iteration']
            kernel = np.ones(shape=(k_size, k_size), dtype=np.uint8)
            result = cv2.erode(binary_image, kernel=kernel, iterations=iteration)
        elif op =='close':
            k_size = self.config['close_k_size']
            iteration = self.config['close_iteration']
            kernel = np.ones(shape=(k_size, k_size), dtype=np.uint8)
            result = cv2.morphologyEx(binary_image, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=iteration)
        elif op =='open':
            k_size = self.config['open_k_size']
            iteration = self.config['open_iteration']
            kernel = np.ones(shape=(k_size, k_size), dtype=np.uint8)
            result = cv2.morphologyEx(binary_image, op=cv2.MORPH_OPEN, kernel=kernel, iterations=iteration)
        return result


    def run_pipeline(self, image):
        """
        Run the pipeline of the image processor
        @param image: the original image with format (h,w,c)
        @return: the preprocessed image with format (h,w) binary image
        """
        grayscale_image = self.convert_to_grayscale(image)
        removed_noise_image = self.remove_noise(grayscale_image)
        binary_image = self.convert_to_binary(removed_noise_image)
        for op in self.config['list_op']:
            binary_image = self.apply_morphological_op(binary_image,op=op)
        return binary_image


if __name__ == '__main__':

    # define files path
    image_path = 'data/test.jpeg'
    config_path = 'config/image_processor.json'

    # load image and parse config
    image = load_image(image_path,gray_scale=False)
    image_processor_config = parse_json(config_path)

    # print(image_processor_config)

    # Test image processor
    ImageProcessor = ImageProcessor(config=image_processor_config)
    result_image = ImageProcessor.run_pipeline(image)
    plot_image(result_image, 'Result_image')

