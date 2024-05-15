""" Parameters of the HDR-Net """
import os

FLAG_FPP = True       # HDR or U-Net

# Phase shift
A       = 130         # A: Average intensity
B       = 90          # B: Modulation intensity
B_min   = 1.5         # B_min: Min value of B
n       = 4           # n: Gray code
gamma   = 1.0         # g: Gamma
N = 12                # MAX number
gray_num = n + 2 + 1  # Image number of gray code

# System parameters
IMG_W   = 1632        # Width of image
IMG_H   = 1248        # Height of image
CAM_W   = 1624        # Width of camera
CAM_H   = 1240        # Height of camera
PRO_W   = 1280        # Width of projector
PRO_H   = 720         # Height of projector

# Calibration
COLS    = 17          # Number of circle(Vertical)
ROWS    = 14          # Number of circle(Horizontal)
DIST    = 6.          # Distance of two circles (mm)

HDR_MAX = 250         # Max value of the HDR algorithm

# Point cloud
LEAF_SIZE = 0.3       # Voxel size
Z_MIN     = -50       # Min distance
Z_MAX     = 100       # Max distance
RADIUS    = 4         # Radius of the radius filter
NB        = 50        # Number of the statistical filter
STDV      = 6.0       # Std of the statistical filter

# Path
IMG_END   = ".jpg"    # Image suffix
ROOT_DIR  = "./"      # Root dir

if FLAG_FPP:
    LOG_DIR = os.path.join(ROOT_DIR, "logs", "FPN")                 # Log directory
    DIR_OUTPUT = os.path.join(ROOT_DIR, "Output", "FPN")            # Output directory
else:
    LOG_DIR = os.path.join(ROOT_DIR, "logs", "UNET")                # Log directory
    DIR_OUTPUT = os.path.join(ROOT_DIR, "Output", "UNET")           # Output directory

DIR_DATA         = os.path.join(ROOT_DIR, "Data")                   # Data
DIR_DATA_Origin  = os.path.join(DIR_DATA, "Data-Origin")            # Origin data
FILE_train       = os.path.join(DIR_DATA, "train.txt")              # Train file
FILE_test        = os.path.join(DIR_DATA, "test.txt")               # Test file
DIR_DATA_pattern = os.path.join(DIR_DATA_Origin, "pattern")         # PS Pattern
DIR_DATA_3d      = os.path.join(DIR_DATA_Origin, "3d")              # 3d reconstruction
DIR_DATA_hdr     = os.path.join(DIR_DATA_Origin, "hdr")             # HDR directory
DIR_DATA_calib   = os.path.join(DIR_DATA_Origin, "calib")           # Calibration
FILE_calib       = os.path.join(DIR_DATA_Origin, "calib_mono.xml")  # Calib file
DIR_DATA_Train   = os.path.join(DIR_DATA, "Data-Train")             # Train directory
DIR_OUTPUT_Phase = os.path.join(DIR_OUTPUT, "Phase")                # Output of phase
DIR_OUTPUT_3d    = os.path.join(DIR_OUTPUT, "3d")                   # Output of 3d reconstruction





