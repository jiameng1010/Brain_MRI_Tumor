from medpy.io import load, save
import nibabel as nib
import numpy as np
import copy

labels_left = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
labels_right = [41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]
LABELS = labels_left + labels_right

def calculate_Dice(vol1, vol2):
    dice = []
    weights1 = []
    for label in LABELS:
        dice.append(calculate_Dice_one(label, vol1, vol2))
        weights1.append(len(np.where(vol1 == label)[0]) + 1)
        # self.weights1.append(1)
        # self.weights2.append(self.weighted_by_distance(label)+1)

    dice = np.asarray(dice)
    weights1 = np.asarray(weights1)
    weights1 = weights1 / np.sum(weights1)
    return dice, weights1

def calculate_Dice_one(label, vol1, vol2):
    from scipy.spatial.distance import dice
    pred = np.zeros_like(vol1)
    pred[np.where(vol1 == label)] = 1
    GT = np.zeros_like(vol2)
    GT[np.where(vol2 == label)] = 1
    intersection = np.logical_and(pred, GT)
    #union = np.logical_or(pred, GT)
    if pred.sum() + GT.sum() == 0:
        return 1.0
    else:
        return 2 * intersection.sum() / (pred.sum() + GT.sum())


def swap_left_right(vol):
    vol_tmp = np.zeros_like(vol)
    for i in range(16):
        vol_tmp[np.where(vol == labels_right[i])] = labels_left[i]
        vol_tmp[np.where(vol == labels_left[i])] = labels_right[i]
    return vol_tmp


class Eval_symmetry():
    def __init__(self, seg_volume):
        self.volume = seg_volume.get_fdata()
        self.eval()

    def get_mean_dice(self, weighted=True):
        if weighted:
            return np.sum(self.weights1 * self.dice)
        else:
            return np.mean(self.dice)

    def eval(self, displacement=None):
        seg_volume_mirrored = np.flip(self.volume, axis=0)
        seg_volume_mirrored = swap_left_right(seg_volume_mirrored)
        dice, weights1 = calculate_Dice(self.volume, seg_volume_mirrored)
        self.weights1 = weights1
        self.dice = dice