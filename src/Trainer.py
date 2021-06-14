import cv2 as cv
import glob
import numpy as np
import datetime
from numpy.linalg import norm
from src.common import clock, mosaic


class KNearest(object):
    def __init__(self, k=3):
        self.k = k
        self.model = cv.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, self.k)
        return results.ravel()

    def load(self, fn):
        self.model = cv.ml.KNearest_load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(object):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv.ml.SVM_RBF)
        self.model.setType(cv.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

    def load(self, fn):
        self.model = cv.ml.SVM_load(fn)

    def save(self, fn):
        self.model.save(fn)


class Trainer:
    def __init__(self, classes, train_paths):
        self.classes = classes
        self.train_paths = train_paths
        self.train_data = []
        self.labels = []
        self.model = None
        self.method = None

    def preprocess_hog(self, imgs):
        samples = []
        for img in imgs:
            gx = cv.Sobel(img, cv.CV_32F, 1, 0)
            gy = cv.Sobel(img, cv.CV_32F, 0, 1)
            height = img.shape[0]
            width = img.shape[1]
            split_index_h = int(height / 2)
            split_index_w = int(width / 2)

            mag, ang = cv.cartToPolar(gx, gy)
            bin_n = 32
            bin = np.int32(bin_n * ang / (2 * np.pi))
            bin_cells = bin[:split_index_w, :split_index_h], bin[split_index_w:, :split_index_h], \
                        bin[:split_index_w, split_index_h:], bin[split_index_w:, split_index_h:]
            mag_cells = mag[:split_index_w, :split_index_h], mag[split_index_w:, :split_index_h], \
                        mag[:split_index_w, split_index_h:], mag[split_index_w:, split_index_h:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)

    def get_data(self, paths):
        i = 0
        datas = []
        labels = []
        for path in paths:
            for filename in glob.glob(path):
                img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
                datas.append(img)
                labels.append(i)
            i = i + 1
        return datas, labels

    def evaluate_model(self, images, samples, labels: np.ndarray):
        resp = self.model.predict(samples)  # type: np.ndarray
        err = (labels != resp).mean()
        print('error: %.2f %%' % (err * 100))

        size = len(self.classes)

        confusion = np.zeros((size, size), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, int(j)] += 1
        print('confusion matrix:')
        print(confusion)

        vis = []
        for img, flag in zip(images, resp == labels):
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if not flag:
                img[..., :2] = 0
            vis.append(img)
        cols = int(len(vis) / 7)
        # cols = 2
        return mosaic(cols, vis)

    def train(self, method):
        train_data, labels = self.get_data(self.train_paths)
        train_data_array = np.array(train_data)
        labels_array = np.array(labels)
        sample_hogs = self.preprocess_hog(train_data_array)
        self.method = method
        print("training " + self.method + "...")
        if method == "knn":
            self.model = KNearest(k=20)
        elif method == "svm":
            self.model = SVM(C=2.67, gamma=5.383)
        self.model.train(sample_hogs, labels_array)
        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print('saving ' + str(self.method) + ' as "models/model_{}_{}.dat"...'.format(self.method, now_time))
        self.model.save("models/model_{}_{}.dat".format(self.method, now_time))
        print("end")

    def test(self, testPaths):
        test_data, labels = self.get_data(testPaths)
        test_data_array = np.array(test_data)
        labels_array = np.array(labels)
        sample_hogs = self.preprocess_hog(test_data_array)
        vis = self.evaluate_model(test_data_array, sample_hogs, labels_array)
        cv.imwrite("result.png", vis)
        cv.imshow(str(self.method) + ' test', vis)

    def evaluate(self, model, img):
        img_array = np.array(img)
        img_hogs = preprocess_single_hog(self, img_array)
        resp = model.predict(img_hogs)
        return resp


def preprocess_single_hog(img):
    # samples = []
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    height = img.shape[0]
    width = img.shape[1]
    split_index_h = int(height / 2)
    split_index_w = int(width / 2)

    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 32
    bin = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bin[:split_index_w, :split_index_h], bin[split_index_w:, :split_index_h], \
                bin[:split_index_w, split_index_h:], bin[split_index_w:, split_index_h:]
    mag_cells = mag[:split_index_w, :split_index_h], mag[split_index_w:, :split_index_h], \
                mag[:split_index_w, split_index_h:], mag[split_index_w:, split_index_h:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    # samples.append(hist)
    return np.float32(hist)
