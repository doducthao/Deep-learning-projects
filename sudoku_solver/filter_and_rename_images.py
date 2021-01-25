import cv2
import os
import shutil
import argparse
import time
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
from main_functions import process, rename_image

if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parse.add_argument('-p', '--filter', action='store_true', help='filter images')
	parse.add_argument('-r', '--rename', action='store_true', help='rename images')
	parse.add_argument('-p1', '--path_1', required=True, help='path to original directory')
	parse.add_argument('-p2', '--path_2', required=True, help='path to destination diretory')
	parse.set_defaults(feature=True)
	args = parse.parse_args()

	if args.filter:
		process(args.path_1)
	if args.rename:
		rename_image(args.path_1, args.path_2)

	

