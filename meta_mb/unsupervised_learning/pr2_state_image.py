import random
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from meta_mb.logger import logger
import pickle
import joblib
from meta_mb.utils.serializable import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from collections import OrderedDict
from meta_mb.unsupervised_learning.vae import VAE

class PR2VAE(object):
    def __init__(self, ):
