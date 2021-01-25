from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
import imutils
import numpy as np 
import argparse 
from main_functions import find_board, extract_digit, get_list_name_img
import os
import shutil
from datetime import datetime
import sys
from train import load_model

def make_raw_data(warped, model, debug, path1, path2):
	length_cell_x = warped.shape[1] // 9
	length_cell_y = warped.shape[0] // 9

	# loop over the grid locations
	for y in range(9):
		for x in range(9):
			begin_x = x*length_cell_x
			end_x = (x+1)*length_cell_x
			begin_y = y*length_cell_y
			end_y = (y+1)*length_cell_y

			# crop the cell from the warped transform image and then extract 
			# the digit from the cell
			cell = warped[begin_y:end_y, begin_x:end_x]
			digit = extract_digit(cell, debug=debug)

			if digit is not None:
				roi = np.array(cv2.resize(digit, (75,75)), dtype=np.float32)
				roi = np.expand_dims(roi, -1)
				roi = np.repeat(roi,3,-1)/255.
				roi = roi.reshape((1,75,75,3))
				
				pred = np.argmax(model.predict(roi), axis=1)[0]
				directory = os.path.join(path2, str(pred))
				path_to_save = os.path.join(directory, str(time())+'.jpg')
				cv2.imwrite(path_to_save, digit)


if __name__ == '__main__':
	parse = argparse.ArgumentParser(description='make data digits 1-9')
	parse.add_argument('-d', '--debug', action='store_true', help='debug!')
	parse.add_argument('-m', '--model', help='path to model trained')
	parse.set_defaults(feature=True)
	args = parse.parse_args()

	path1 = 'boards/'
	path2 = 'data_raw/'
	model = load_best_model(args.model)
	debug = args.debug
	if not os.path.exists(path1):
		os.mkdir(path1)
	if not os.path.exists(path2):
		os.mkdir(path2)
		for i in range(1,10):
			os.mkdir(os.path.join(path2, str(i)))

	list_name_img = get_list_name_img(path1)
	list_name_img = [path1+i for i in list_name_img]
	for img_name in list_name_img:
		img = cv2.imread(img_name)
		warped = find_board(img, debug=debug)[1]
		make_raw_data(warped, model, debug, path1, path2)







