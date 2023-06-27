from NeuroGraph import utils
import numpy as np
from nilearn.image import load_img



img = load_img("NeuroGraph/data/raw/1.nii.gz")
regs = np.loadtxt("NeuroGraph/data/raw/1.txt")
fmri = img.get_fdata()
fc = utils.preprocess(fmri, regs,100)
# fc = np.load("NeuroGraph/data/fc.npy")
print(fc.shape) 
data = utils.construct_data(fc, 1)
print(data)