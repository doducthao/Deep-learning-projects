from train import load_best_model
from sudoku import Sudoku

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
import imutils
import numpy as np 
import argparse 
import shutil
import os
import time
from main_functions import find_board, detect_board, extract_digit

if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parse.add_argument('-i', '--image', help='Image to solve')
	parse.add_argument('-f', '--folder', help='Folder contain images')
	parse.add_argument('-d', '--debug', dest='debug', action='store_true', help='Debug code')
	parse.add_argument("-m", "--model", required=True, help="Model training")

	parse.add_argument('-o', '--one', action='store_true')
	parse.add_argument('-ma', '--many', action='store_true')
	parse.set_defaults(feature=True)
	args = parse.parse_args()

	path_solution = 'data/solutions'
	if not os.path.exists(path_solution):
		os.makedirs(path_solution)

	print('Loading model to classify digits...')
	model = load_best_model(args.model)

	if args.many:
		print('Folder has {} images'.format(len(os.listdir(args.folder))))
		for img in sorted(os.listdir(args.folder)):
			name_img = img
			print('Processing the image {}'.format(img))
			img = cv2.imread(os.path.join(args.folder,img))
			# print('The image has shape {}'.format(img.shape))
			img = imutils.resize(img, width=600)
			if args.debug:
				print('The code has debugs!')
			puzzle_img, warped = find_board(img, debug=args.debug)

			cell_locs, board = detect_board(warped, model, debug=args.debug)

			print('OCR Sudoku board')
			puzzle = Sudoku(3,3, board=board.tolist())
			puzzle.show()

			print('Soving Sudoku puzzle...')
			result = puzzle.solve()
			result.show_full()

			# loop over the cell locations and board
			for (cell_row, board_row_result, board_row) in zip(cell_locs, result.board, board):
				# loop over individual cell in the row
				for (box, digit_result, digit_board) in zip(cell_row, board_row_result, board_row):
					if digit_board == 0:
					# unpack the cell coordinates
						begin_x, begin_y, end_x, end_y = box
						# compute the coordinates of where the digit will be drawn
						# on the output puzzle image
						text_x = int((end_x - begin_x) * 0.33)
						text_y = int((end_y - begin_y) * -0.2)
						text_x += begin_x
						text_y += end_y
						# draw the result digit on the Sudoku puzzle image
						cv2.putText(puzzle_img, str(digit_result), (text_x, text_y),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv2.imwrite(os.path.join(path_solution,'SolutionOf_'+name_img), puzzle_img)

	if args.one:
		name_img = args.image.split('/')[-1]
		img = cv2.imread(args.image)
		# print('The image has shape {}'.format(img.shape))
		img = imutils.resize(img, width=600)
		if args.debug:
			print('The code has debugs!')
		puzzle_img, warped = find_board(img, debug=args.debug)

		cell_locs, board = detect_board(warped, model, debug=args.debug)

		print('OCR Sudoku board')
		puzzle = Sudoku(3,3, board=board.tolist())
		puzzle.show()

		print('Soving Sudoku puzzle...')
		result = puzzle.solve()
		result.show_full()
		# print(result.board)

		# loop over the cell locations and board
		for (cell_row, board_row_result, board_row) in zip(cell_locs, result.board, board):
			# loop over individual cell in the row
			for (box, digit_result, digit_board) in zip(cell_row, board_row_result, board_row):
				if digit_board == 0:
				# unpack the cell coordinates
					begin_x, begin_y, end_x, end_y = box
					# compute the coordinates of where the digit will be drawn
					# on the output puzzle image
					text_x = int((end_x - begin_x) * 0.33)
					text_y = int((end_y - begin_y) * -0.2)
					text_x += begin_x
					text_y += end_y
					# draw the result digit on the Sudoku puzzle image
					cv2.putText(puzzle_img, str(digit_result), (text_x, text_y),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.imwrite(os.path.join(path_solution,'SolutionOf_'+name_img), puzzle_img)