from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
import imutils
import numpy as np 
import argparse 
import shutil
import os
import time
import re
from datetime import datetime
import skimage 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def get_list_name_img(path):
	list_img_path = os.listdir(path)
	return list_img_path

def process_duplicate(path):
	list_img_path = get_list_name_img(path)
	n = len(list_img_path)
	list_img_path = [path+i for i in list_img_path]

	list_shape = []
	list_remove = []

	for img_path in list_img_path:
		img = cv2.imread(img_path)

		if img.shape not in list_shape:
			list_shape.append(img.shape)
		else:
			list_remove.append(img_path)
	if len(list_remove) == 0:
		print('Duplicate images not found!')
	else:
		for img in list_remove:
			os.remove(img)
	print('Original images number:', n)
	print('Later images number:', len(os.listdir(path)))

def rename_image(path):
	print('Renaming images files...')
		
	list_img_path = get_list_name_img(path)
	for i, img_name in enumerate(sorted(list_img_path)):
		tail = img_name.rsplit('.', 2)[-1]
		img_name_new = str(datetime.now())+'.'+tail
		img_name_new_2 = str(i+1)+'.'+tail
		try:
			os.rename(os.path.join(path,img_name), os.path.join(path,img_name_new))
			os.rename(os.path.join(path, img_name_new), os.path.join(path, img_name_new_2))
		except:
			pass
	print('Done!')

def split_data(path, rate):
	data = []
	for root, dirs, files in sorted(os.walk(path)):
		if len(files)==0:
			continue
		n = int(len(files)*rate)
		train_data_path = files[:n]
		validation_data_path = files[n:]
		data.append((root, train_data_path, validation_data_path))
	return data

def compress_to_npy(path, train=False, validation=False):
	data_x, data_y = [], []
	if train:
		path = os.path.join(path, 'train')
	if validation:
		path = os.path.join(path, 'validation')
	for root, dirs, files in sorted(os.walk(path), key=lambda x: x[0]):
		if len(files) == 0:
			continue
		name_class = root.split('/')[-1]
		for img in files:
			img = os.path.join(root, img)
			img = cv2.imread(img, 0)
			img = cv2.resize(img, (28,28))

			data_x.append(img)
			data_y.append(int(name_class))	

	data_x = np.array(data_x, dtype=np.uint8)
	data_y = np.array(data_y, dtype=np.uint8)
	
	if train:
		np.save(os.path.join(path, 'train_data.npy'), data_x)
		np.save(os.path.join(path, 'train_labels.npy'), data_y)
		print('Save file .npy in {}'.format(path))
	if validation:
		np.save(os.path.join(path, 'validation_data.npy'), data_x)
		np.save(os.path.join(path, 'validation_labels.npy'), data_y)
		print('Save file npy in {}'.format(path))
	
def transform_binary_color(path1, path2):
	if not os.path.exists(path2):
		os.makedirs(path2)
		print('Create ',path2)
	else:
		print('{} has existed'.format(path2))

	for root, dirs, files in os.walk(path1):
		if len(files) == 0:
			continue
		for path_img in files:
			img = os.path.join(root, path_img)
			img = cv2.imread(img)
			img = cv2.bitwise_not(img)
			cv2.imwrite(os.path.join(path2, path_img), img)
	print('Done')

def filter_data(path1, path2):
	for i in range(10):
		sub_folder = os.path.join(path2, str(i))
		try:
			if not os.path.exists(sub_folder):
				os.makedirs(sub_folder)
				print('Create successful')
			else:
				print('Folder has existed')
		except:
			print('Check again')		

	for root, dirs, files in sorted(os.walk(path1)):
		if len(files) == 0:
			continue
		count=0
		class_name = root.split('/')[-1]
		for img in files:
			extension = img.split('.')[-1]
			path_img = os.path.join(root, img)
			img = mpimg.imread(path_img)
			if img.shape[0] < 28  or img.shape[1] < 28:
				continue
			count+=1
			new_path = os.path.join(path2, class_name, str(datetime.now())+'_'+str(count)+'.'+extension)
			shutil.copy(path_img, new_path)
		print('Found {} images in class {} that satisfy the requirements'.format(count, int(class_name)))


	
def make_data_train_validation(path_1, path_2, rate):
	folder_train = os.path.join(path_2, 'train')
	folder_validation = os.path.join(path_2, 'validation')
	for i in range(10):
		sub_folder_train = os.path.join(folder_train, str(i))
		sub_folder_validation = os.path.join(folder_validation, str(i))
		try:
			if not os.path.exists(sub_folder_train):
				os.makedirs(sub_folder_train)

			if not os.path.exists(sub_folder_validation):
				os.makedirs(sub_folder_validation)
		except:
			print('Check again!')
		else:
			print('Create successful!')

	data = split_data(path_1, rate)
	for i, sub_data in enumerate(data):
		name_folder, train_data_path, validation_data_path = sub_data
		for path in train_data_path:
			shutil.copy(os.path.join(name_folder, path), os.path.join(folder_train, str(i), path))

		for path in validation_data_path:
			shutil.copy(os.path.join(name_folder, path), os.path.join(folder_validation, str(i), path))

def find_board(img, debug=False):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (9,9), 0)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
	puzzle_cnt = None

	for c in cnts:
		p = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*p, True)
		if len(approx) == 4:
			puzzle_cnt = approx
			break	

	if puzzle_cnt is None:
		raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

	if debug:
	# draw the contour of the puzzle on the image and then display
	# it to our screen for visualization/debugging purposes
		output = img.copy()
		cv2.drawContours(output, [puzzle_cnt], -1, (0, 255, 0), 3)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)

	# apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
	puzzle = four_point_transform(img, puzzle_cnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))
	# check to see if we are visualizing the perspective transform
	if debug:
		# show the output warped image (again, for debugging purposes)
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.waitKey(0)
	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)

def extract_digit(cell, debug=False):
	# cell: an ROI representing an individual cell of the Sudoku puzzle
	# (may or may not contain a digit)
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
	thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)
	if debug:
		cv2.imshow('cell thresh', thresh)
		cv2.waitKey(0)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# print(len(cnts))

	# if no contours were found than this is an empty cell
	if len(cnts) == 0:
		return None
	# otherwise, find the largest contour in the cell and create a
	# mask for the contour
	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)
	# cv2.imshow('masked', mask)
	# cv2.waitKey(0)

	# compute the percentage of masked pixels relative to the total
	# area of the image
	(h, w) = thresh.shape
	percent_filled = cv2.countNonZero(mask) / float(w * h)

	# if less than 3% of the mask is filled then we are looking at
	# noise and can safely ignore the contour
	if percent_filled < 0.03:
		return None

	# apply the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)

	# check to see if we should visualize the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)
	# return the digit to the calling function
	return digit

def detect_board(warped, model, debug):
	# initialize sudoku 9x9 board
	board = np.zeros((9,9), dtype='int')
	# infer the location of each cell by dividing the warped image into a 9x9 grid
	length_cell_x = warped.shape[1] // 9
	length_cell_y = warped.shape[0] // 9
	cell_locs = []
	# loop over the grid locations
	for y in range(9):
		row = []
		for x in range(9):
			begin_x = x*length_cell_x
			end_x = (x+1)*length_cell_x
			begin_y = y*length_cell_y
			end_y = (y+1)*length_cell_y
			row.append((begin_x, begin_y, end_x, end_y))
			# crop the cell from the warped transform image and then extract 
			# the digit from the cell
			cell = warped[begin_y:end_y, begin_x:end_x]
			digit = extract_digit(cell, debug=debug)

			if digit is not None:
				roi = np.array(cv2.resize(digit, (75,75)), dtype=np.float32)
				# roi = skimage.color.gray2rgb(roi)
				# print(roi.shape)
				if debug:
					cv2.imshow('roi', roi)
					cv2.waitKey(0)
				roi = np.expand_dims(roi, -1)
				roi = np.repeat(roi,3,-1)/255.
				roi = roi.reshape((1,75,75,3))
				pred = np.argmax(model.predict(roi), axis=1)[0]
				board[y,x] = pred 

		cell_locs.append(row)
	return (cell_locs, board)

if __name__ == '__main__':
	pass




