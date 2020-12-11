
import sys, os
import shutil
import subprocess
import multiprocessing
from datetime import datetime
import time
import pickle
import glob
import copy
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import pytraj as pt
import parmed as pmd
import waterlib as wl
from pymbar import mbar
from genetic_lib import calcdGsolv

try:
  thistime = int(sys.argv[1])
except IndexError:
  thistime = -1

defaultSimDirs = ['Quad_0.25X_0.25Y', 'Quad_0.25X_0.75Y', 'Quad_0.75X_0.25Y', 'Quad_0.75X_0.75Y']
defaultKXY = [10.0, 10.0, 10.0, 10.0]
defaultRefX = [7.4550, 7.4550, 22.3650, 22.3650]
defaultRefY = [8.6083, 25.8249, 8.6083, 25.8249]
defaultDistRefX = [7.4550, 7.4550, 7.4550, 7.4550]
defaultDistRefY = [8.6083, 8.6083, 8.6083, 8.6083]
numLambdaStates = 19 #Change this to 25 for acetic acid (or any other solutes added later with different number of states)
                     #This is also hard-coded in the genetic algorithm, so cannot run that if have different number of states

(thisMetric) = calcdGsolv(defaultSimDirs, defaultKXY, defaultRefX, defaultRefY, defaultDistRefX, defaultDistRefY, numLambdaStates, endTime=thistime, kB=0.008314459848, T=298.15, verbose=True)

print("  dGsolv = %f" % thisMetric[0])
print("  dGsolvError = %f" % thisMetric[1])
print("  Samples at all states: %s" % str(thisMetric[2].tolist()))
print("  First state total samples = %i" % thisMetric[3])
print("  Last state total samples = %i" % thisMetric[4])
print("  First state min samples in XY bin = %f" % thisMetric[5])
print("  First state max samples in XY bin = %f" % thisMetric[6])
print("  Last state min samples in XY bin = %f" % thisMetric[7])
print("  Last state max samples in XY bin = %f" % thisMetric[8])


