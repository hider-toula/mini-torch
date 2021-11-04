import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.module import Linear,TanH,Sigmoide
from nn.loss import MSELoss
from tools.basic import *
from tools.nntools import Sequentiel,Optim
import matplotlib.pyplot as plt
