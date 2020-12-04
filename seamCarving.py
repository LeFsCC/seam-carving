import cv2
from tqdm import trange
import numpy as np
from scipy.ndimage.filters import convolve
from hog import HogEnergy


class SeamCarving:
    def __init__(self):
        self.filter_du = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])

        self.filter_du = np.stack([self.filter_du] * 3, axis=2)

        self.filter_dv = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])

        self.filter_dv = np.stack([self.filter_dv] * 3, axis=2)
        self.all_time = 0
        self.now_time = 0
        self.now_image = 0
        self.end_mark = False
        self.dire = "r"
        self.energy_select = "e1"

    def carve(self, img, s, d, e):
        self.energy_select = e
        self.now_image = img
        self.all_time = 0
        self.now_time = 0
        self.end_mark = False
        self.dire = d
        if self.dire == "r":
            res = self.row(img, s)
        elif self.dire == "c":
            res = self.column(img, s)
        else:
            return None
        return res

    def energy_e1(self, img):
        img = img.astype('float32')
        convolved = np.absolute(convolve(img, self.filter_du)) + np.absolute(convolve(img, self.filter_dv))
        return convolved.sum(axis=2)

    def laplacian(self, img):
        kernel_l = np.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        kernel_l = np.stack([kernel_l] * 3, axis=2)
        img = img.astype('float32')
        lap = np.absolute(convolve(img, kernel_l))
        return lap.sum(axis=2)

    def HOG(self, img):
        b, g, r = cv2.split(img)
        b_energy = np.sqrt(np.absolute(cv2.Scharr(b, -1, 1, 0)) ** 2 + np.absolute(cv2.Scharr(b, -1, 0, 1)) ** 2)
        g_energy = np.sqrt(np.absolute(cv2.Scharr(g, -1, 1, 0)) ** 2 + np.absolute(cv2.Scharr(g, -1, 0, 1)) ** 2)
        r_energy = np.sqrt(np.absolute(cv2.Scharr(r, -1, 1, 0)) ** 2 + np.absolute(cv2.Scharr(r, -1, 0, 1)) ** 2)
        E = b_energy + g_energy + r_energy
        hog = HogEnergy(img, cell_size=11, bin_size=8)
        histogram = hog.extract()
        histogram = histogram / np.linalg.norm(histogram)
        return E / histogram

    def get_now_image(self):
        return self.now_image

    def forward_energy(self, im):
        h, w = im.shape[:2]
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

        energy = np.zeros((h, w))
        m = np.zeros((h, w))

        U = np.roll(im, 1, axis=0)
        L = np.roll(im, 1, axis=1)
        R = np.roll(im, -1, axis=1)

        cU = np.abs(R - L)
        cL = np.abs(U - L) + cU
        cR = np.abs(U - R) + cU

        for i in range(1, h):
            mU = m[i - 1]
            mL = np.roll(mU, 1)
            mR = np.roll(mU, -1)

            mULR = np.array([mU, mL, mR])
            cULR = np.array([cU[i], cL[i], cR[i]])
            mULR += cULR

            argmins = np.argmin(mULR, axis=0)
            m[i] = np.choose(argmins, mULR)
            energy[i] = np.choose(argmins, cULR)

        return energy

    def get_all_time(self, img, scale_c):
        r, c, _ = img.shape
        new_c = int(scale_c * c)
        if scale_c < 1.0:
            self.all_time = c - new_c
        else:
            self.all_time = new_c - c
        return self.all_time

    def get_now_time(self):
        return self.now_time

    def get_end_mark(self):
        return self.end_mark

    def column(self, img, scale_c):
        r, c, _ = img.shape
        new_c = int(scale_c * c)
        if scale_c < 1.0:
            print("reduce image")
            for i in trange(c - new_c):
                self.now_time = i
                img = self.delete_seam(img)
                self.now_image = img
            self.now_time += 1
        else:
            print("enlarge image")
            for i in trange(new_c - c):
                self.now_time = i
                img = self.add_seam(img)
                self.now_image = img
            r, c, x = img.shape
            self.now_time += 1
            for i in range(r):
                for j in range(c):
                    if (img[i][j] == [0, 0, 0]).all():
                        img[i][j] = img[i][j - 1]
            self.now_image = img
        self.end_mark = True
        return img

    def row(self, img, scale_r):
        img = np.rot90(img, 1, (0, 1))
        img = self.column(img, scale_r)
        img = np.rot90(img, 3, (0, 1))
        self.now_image = img
        self.end_mark = True
        return img

    def delete_seam(self, img):
        r, c, _ = img.shape

        M, backtrack = self.compute_backward_cost(img)
        mask = np.ones((r, c), dtype=np.bool)

        j = np.argmin(M[-1])
        for i in reversed(range(r)):
            mask[i, j] = False
            j = backtrack[i, j]

        mask = np.stack([mask] * 3, axis=2)
        img = img[mask].reshape((r, c - 1, 3))
        return img

    def add_seam(self, img):
        r, c, x = img.shape
        M, backtrack = self.compute_backward_cost(img)

        output = np.zeros((r, c + 1, 3))
        j = np.argmin(M[-1])
        for i in reversed(range(r)):
            for ch in range(3):
                output[i, : j, ch] = img[i, : j, ch]
                output[i, j + 1:, ch] = img[i, j:, ch]
            j = backtrack[i, j]

        return output

    def compute_backward_cost(self, img):
        r, c, _ = img.shape
        print(self.energy_select)
        if self.energy_select == "forward":
            energy_map = self.forward_energy(img)
        elif self.energy_select == "e1":
            energy_map = self.energy_e1(img)
        elif self.energy_select == "laplacian":
            energy_map = self.laplacian(img)
        elif self.energy_select == "hog":
            energy_map = self.HOG(img)
        else:
            return None, None

        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            for j in range(0, c):
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        return M, backtrack

