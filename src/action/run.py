import os
import sys

srcPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(srcPath)
from model.AutoEncoder import AutoEncoder
from model.WaterReconstruct import WaterReconstruct

rootPath = os.path.dirname(srcPath)
modelPath = rootPath + "/model/auto_encoder"
model = WaterReconstruct()
# model.BuildData(rootPath + "/image/mark/taptap_small.png", rootPath + "/image/source/main")
model.Train(10000000)
