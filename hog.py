import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class HogEnergy:

    def __init__(self, img, cell_size=11, bin_size=8):
        self.img = img
        if len(img.shape) > 2:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    def extract(self):
        height, width = self.img.shape

        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros(
            (math.ceil(height / self.cell_size), math.ceil(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)
        for i in range(cell_gradient_vector.shape[0] - 1):
            cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                             -self.cell_size:]
            cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                         -self.cell_size:]
            cell_gradient_vector[i][cell_gradient_vector.shape[1] - 1] = self.cell_gradient(cell_magnitude, cell_angle)
        for j in range(cell_gradient_vector.shape[1] - 1):
            cell_magnitude = gradient_magnitude[-self.cell_size:,
                             j * self.cell_size:(j + 1) * self.cell_size]
            cell_angle = gradient_angle[-self.cell_size:,
                         j * self.cell_size:(j + 1) * self.cell_size]
            cell_gradient_vector[cell_gradient_vector.shape[0] - 1][j] = self.cell_gradient(cell_magnitude, cell_angle)
        cell_magnitude = gradient_magnitude[-self.cell_size:,
                         -self.cell_size:]
        cell_angle = gradient_angle[-self.cell_size:,
                     -self.cell_size:]
        cell_gradient_vector[cell_gradient_vector.shape[0] - 1][cell_gradient_vector.shape[1] - 1] = self.cell_gradient(
            cell_magnitude, cell_angle)

        done_before = np.zeros([cell_gradient_vector.shape[0], cell_gradient_vector.shape[1]], bool)
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])

                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)

                if magnitude != 0:
                    if not done_before[i][j]:
                        cell_gradient_vector[i][j] = cell_gradient_vector[i][j] / magnitude
                        done_before[i][j] = True
                    if not done_before[i][j + 1]:
                        cell_gradient_vector[i][j + 1] = cell_gradient_vector[i][j + 1] / magnitude
                        done_before[i][j + 1] = True
                    if not done_before[i + 1][j]:
                        cell_gradient_vector[i + 1][j] = cell_gradient_vector[i + 1][j] / magnitude
                        done_before[i + 1][j] = True
                    if not done_before[i + 1][j + 1]:
                        cell_gradient_vector[i + 1][j + 1] = cell_gradient_vector[i + 1][j + 1] / magnitude
                        done_before[i + 1][j + 1] = True

        hog_pixel = self.get_pixel_max(cell_gradient_vector)

        return hog_pixel

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))

        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

    def get_pixel_max(self, cell_gradient):
        image = np.zeros_like(self.img)
        for x in range(cell_gradient.shape[0] - 1):
            for y in range(cell_gradient.shape[1] - 1):
                cell_grad = cell_gradient[x][y]
                max_hog = cell_grad.max()
                for height in range(x * self.cell_size, (x + 1) * self.cell_size):
                    for width in range(y * self.cell_size, (y + 1) * self.cell_size):
                        image[height][width] = max_hog

        for x in range(cell_gradient.shape[0] - 1):
            cell_grad = cell_gradient[x][cell_gradient.shape[1] - 1]
            max_hog = cell_grad.max()
            for height in range(x * self.cell_size, (x + 1) * self.cell_size):
                for width in range(y * self.cell_size, self.img.shape[1]):
                    image[height][width] = max_hog

        for y in range(cell_gradient.shape[1] - 1):
            cell_grad = cell_gradient[cell_gradient.shape[0] - 1][y]
            max_hog = cell_grad.max()
            for height in range(x * self.cell_size, self.img.shape[0]):
                for width in range(y * self.cell_size, (y + 1) * self.cell_size):
                    image[height][width] = max_hog

        cell_grad = cell_gradient[cell_gradient.shape[0] - 1][cell_gradient.shape[1] - 1]
        max_hog = cell_grad.max()
        for height in range(self.img.shape[0] - self.img.shape[0] % self.cell_size, self.img.shape[0]):
            for width in range(self.img.shape[1] - self.img.shape[1] % self.cell_size, self.img.shape[1]):
                image[height][width] = max_hog

        return image
