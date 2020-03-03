from preprocess_image import ImageProcessor
from utils.utils import *
import operator
import logging
import torch
from cnn import CNNModel
import cv2
import time
from cnn import predict_on_images
import numpy as np
from sudoku_solver import solve_sudoku
from keras_model import loadmodel,getvaliddatagen,predict_images
logging.basicConfig(level=logging.INFO)
model = loadmodel("pretrained/model.pkl")
validata = getvaliddatagen('pretrained/validdatagen.pkl')
class Sudoku:

    def __init__(self,sudoku_config,preprocess_config):
        """
        @param preprocess_config: params config of the image preprocessor
        """
        self.image_processor = ImageProcessor(config=preprocess_config)
        self.sudoku_config = sudoku_config

    def get_distance_bettween_two_points(self,point1,point2):
        """
        Get distance between two points
        @param point1: the numpy array with format [x,y]
        @param point2: the numpy array with format [x,y]
        @return: a float denotes the distance between two points
        """
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        return np.sqrt((delta_x**2) + (delta_y**2))

    def get_coordinates_of_largest_contour(self,image):
        """
        Get the coordinates of the largest contours in image (top_left,top_right,bottom_left,bottom_right)
        @param image: the rgb image with format (h,w,c)
        @return: the top_left,top_right,bottom_left,bottom_right of the largest contours
        """
        image_copy = image.copy()
        processed_image = self.image_processor.run_pipeline(image_copy)
        contours, h = cv2.findContours(processed_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=cv2.contourArea,reverse=True)
        polygon = contours[0]
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        print("\n"+str(bottom_right)+" "+str(bottom_left)+" "+str(top_right)+" "+str(top_left))
        coordinates = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
        print(coordinates)
        return coordinates

    def crop_and_wrap_sudoku_roi(self,image,crop_rect):
        """
        Crop and wrap the sudoku roi from the original image
        @param image: the original image with format (h,w,c)
        @param crop_rect: the coordinates of the sudoku roi in the original image
        @return: the wrapped and cropped sudoku roi
        """
        top_left, top_right, bottom_right, bottom_left = crop_rect

        src = np.array([top_left,top_right,bottom_right,bottom_left],dtype=np.float32)
        side = max([
            self.get_distance_bettween_two_points(bottom_right,top_right),
            self.get_distance_bettween_two_points(top_left,bottom_left),
            self.get_distance_bettween_two_points(bottom_right,bottom_left),
            self.get_distance_bettween_two_points(top_left,top_right)
        ])
        # plot_image(image,'original_image')
        dst = np.array([[0,0],[0,side-1],[side -1,side -1], [side-1,0]],dtype=np.float32)
        m = cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(image,m,(int(side),int(side)))

    def crop_from_rect(self, image , crop_rect):
        """
        crop roi (region of interest) from the original image
        @param image: the original image with format (h,w,c)
        @param crop_rect: the coordinates of the roi (top_left, bottom right)
        @return:
        """
        return image[int(crop_rect[0][1]):int(crop_rect[1][1]),int(crop_rect[0][0]):int(crop_rect[1][0])]

    def pass_condition(self,stats,w,h):
        """
        check the condition of the roi number
        @param stats: infomation about the connected components
        @param w: the width of the image contains the connected component
        @param h: the height of the image contains the connected component
        @return: True if the connected component pass the condition else false
        """
        start_left = stats[0]
        start_top = stats[1]
        stats_width = stats[2]
        stats_height = stats[3]
        stats_area = stats[4]
        # print('thong tin: ',stats)
        if stats_width > 0.7 * w or stats_height > 0.7 * h:
            return False
        if stats_width < 4 or stats_height < 0.1 * h:
            return False
        if start_left < 0.05 * w:
            return False
        if stats_area < 0.04 * w * h:
            return False
        return True

    def __find_largest_feature(self,digit_roi):
        """
        Get the largest connected component passes the condition of number roi in the image
        @param digit_roi: the roi contain the number and noise
        @return: the mask of the digit
        """
        image_copy = digit_roi.copy()
        # plot_image(image_copy,'roi')
        h,w = image_copy.shape[:2]
        binary_image = self.image_processor.convert_to_binary(image_copy)
        # print(np.unique(binary_image))
        label,stats,centroids = cv2.connectedComponentsWithStats(binary_image,4,cv2.CV_32S)[1:]
        # print(stats)
        seed_point = None
        max_area = 10
        # print(stats)
        target_label = None
        for i in np.unique(label):
            area = stats[i,cv2.CC_STAT_AREA]
            if area >= max_area:
                centroid = centroids[i]
                if self.pass_condition(stats[i],w,h):
                    seed_point = centroid
                    max_area = area
                    target_label = i
        mask = np.zeros_like(label,dtype=np.uint8)
        if target_label is not None:
            mask[label==target_label] = 255
            mask = self.image_processor.apply_morphological_op(mask,op='close')
            # mask = self.image_processor.apply_morphological_op(mask, op='open')
            # plot_image(mask,'label')
        return mask,target_label

    def __pad_to_central(self,digit_roi):
        """
        pad for centring the number on the image
        @param digit_roi: the number roi with format (h,w)
        @return: padded roi of the input image
        """
        h,w = digit_roi.shape
        min_left = 1000
        min_top = 1000
        max_right = -1
        max_bottom = -1
        for i in range(h):
            for j in range(w):
                if digit_roi[i,j] != 0:
                    if j <= min_left:
                        min_left = j
                    if j >= max_right:
                        max_right = j
                    if i <= min_top:
                        min_top = i
                    if i >= max_bottom:
                        max_bottom = i
        padded_roi = digit_roi
        if min_left < w - max_right:
            if min_top < h - max_bottom:
                pad_width = w - max_right - min_left
                pad_height = h - max_bottom - min_top
                padded_roi = np.pad(padded_roi,pad_width=((pad_height,0),(pad_width,0)),constant_values=0)
            elif min_top > h - max_bottom:
                pad_width = w - max_right - min_left
                pad_height = min_top - (h-max_bottom)
                padded_roi = np.pad(padded_roi, pad_width=((0,pad_height), (pad_width, 0)), constant_values=0)
        elif min_left > w - max_right:
            if min_top < h - max_bottom:
                pad_width = min_left - (w - max_right)
                pad_height = h - max_bottom - min_top
                padded_roi = np.pad(padded_roi,pad_width=((pad_height,0),(pad_width,0)),constant_values=0)
            elif min_top > h - max_bottom:
                pad_width = min_left - (w-max_right)
                pad_height = min_top - (h-max_bottom)
                padded_roi = np.pad(padded_roi, pad_width=((0,pad_height), (pad_width, 0)), constant_values=0)
        # plot_image(padded_roi,'digit')
        return padded_roi


    def extract_one_digit(self, sudoku_roi, crop_rect):
        """
        Extract the mask of one digit on sudoku roi with its coordinates
        @param sudoku_roi: the sudoku roi with format (h,w)
        @param crop_rect: the coordinates of top left and bottom right of the digit roi
        @return: the mask contain the digit
        """
        digit = self.crop_from_rect(sudoku_roi,crop_rect)
        mask,target_label = self.__find_largest_feature(digit)
        return mask,target_label

    def get_digit_coordinates_from_sudoku_roi(self,sudoku_roi):
        """
        Get 81 gird cell from sudoku roi
        @param sudoku_roi: the rgb image with format (h,w,c)
        @return: 81 grid cell from the sudoku roi
        """
        squares = []
        side = sudoku_roi.shape[:-1]
        side = side[0]/9
        for j in range(9):
            for i in range(9):
                p1 = (i*side,j*side)
                p2 = ((i+1)*side,(j+1)*side)
                squares.append((p1,p2))
        return squares

    def run_pipeline(self,image):
        """
        Run pipeline for sudoku solver
        @param image: the sudoku image
        @return: solution
        """
        image_copy = image.copy()
        grayscale = self.image_processor.convert_to_grayscale(image_copy)
        crop_rect = self.get_coordinates_of_largest_contour(image_copy)
        sudoku_roi = self.crop_and_wrap_sudoku_roi(grayscale,crop_rect)
        plot_image(sudoku_roi,'roi')
        squares = self.get_digit_coordinates_from_sudoku_roi(sudoku_roi)
        mask_digits = []
        labels = []
        mask_dict = dict()
        for i in range(len(squares)):
            mask,target_label = self.extract_one_digit(sudoku_roi,squares[i])
            labels.append(target_label)
            if target_label is not None:
                mask = self.__pad_to_central(mask)
                mask_digits.append(mask)
                mask_dict[str(i)] = mask
        map = np.zeros(shape=(1, 81))
        if len(mask_digits) > 0:
            mask_dict = {k:cv2.resize(x,(28,28)) for k,x in mask_dict.items()}
            # plot_images(mask_digits,'digits')
            mask_digits = np.array(mask_digits)
            result_dict = predict_images(mask_dict,model,validata)
            for k,v in result_dict.items():
                map[0][int(k)] = v
        map = map.reshape(9,9)
        can_solve = solve_sudoku(map)
        if can_solve:
            print(map)
            display_solution(map)
        else:
            print("no solution")

if __name__ == '__main__':
    image_path = 'data/test3.png'
    image_processor_config_path = 'config/image_processor.json'
    sudoku_config_path = 'config/sudoku.json'
    image = load_image(image_path, gray_scale=False)
    image_processor_config = parse_json(image_processor_config_path)
    sudoku_config = parse_json(sudoku_config_path)
    sudoku = Sudoku(preprocess_config=image_processor_config, sudoku_config=sudoku_config)
    sudoku.run_pipeline(image)


