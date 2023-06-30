from NeuroGraph import utils
import numpy as np
from nilearn.image import load_img
import sys


def main(file, roi,method):

    img = load_img("NeuroGraph/data/raw/1.nii.gz")
    regs = np.loadtxt("NeuroGraph/data/raw/1.txt")
    fmri = img.get_fdata()
    Y = utils.parcellation(fmri,100)
    Y = utils.remove_drifts(Y)
    print(Y.shape)
    M = utils.regress_head_motions(Y, regs)
    print(M.shape)
    fc = utils.construct_corr(M)

    np.save(fc)
    # fc = np.load("NeuroGraph/data/fc.npy")
    print(fc.shape) 
    data = utils.construct_data(fc, 1)
    print(data)


file_path = sys.argv[1]


