from PIL import Image
import numpy as np
import scipy.io


def printLabelArray(label):
    for i in range(label.shape[0]):
        outstr = ''
        for j in range(label.shape[1]):
            outstr += str(label[i][j]) + ' '
        print(outstr)


def load_image(imname, mu, return_raw=False):
    img = Image.open(imname).convert("RGB")
    im = np.array(img, dtype=np.float32)  # shape = [H, W, 3]
    im = im[:, :, ::-1]  # rgb -> bgr
    im -= mu  # subtract mean
    im = np.expand_dims(im, axis=0)  # shape = [1, H, W, 3]

    if return_raw:
        im_raw = np.array(img, dtype=np.uint8)  # shape = [H, W, 3]
        return im, im_raw
    else:
        return im


def load_label(label_path_):
    label = scipy.io.loadmat(label_path_)['CLASS_GT']
    label = np.array(label, dtype=np.uint8)  # shape = [H, W]
    label = label[np.newaxis, ...]  # shape = [1, H, W]
    return label