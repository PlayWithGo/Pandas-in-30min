#**************************************************************************************************#
#                                               Deep Learning Developing Kit                                                    #
#								        		 	          Json Handler   	                                                           #
#                                                   www.tianshicangxie.com                                                         #
#                                      Copyright © 2015-2018 Celestial Tech Inc.                                           #
#**************************************************************************************************#

# modules
import os 
import sys
import numpy as np
import json

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..\\..'))

sys.path.append(str(dir_path + '\\src\\Json')) 
from JsonParser import VectorParser


if __name__