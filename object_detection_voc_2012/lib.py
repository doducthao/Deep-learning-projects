import os
import urllib.request
import zipfile
import tarfile
import random
import xml.etree.ElementTree as ET 
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from math import sqrt
import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Function
import torch.nn.init as init

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
