import pandas as pd
import glob
import matplotlib.pyplot as plt
import mne 
import scipy.signal as sps
import scipy.fftpack as spf
import numpy as np
import pywt
import seaborn as sns
import scipy as sp
import time
import sys

class0=   'Normal'
Class1 =  'Sick'
print('hlw guys!!')
def till_display(data):
    print(data, file=sys.stderr)
    time.sleep(3)
    return "hi chali!!"

def till_classification(file_address):
    print(file_address)