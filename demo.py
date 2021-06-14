import argparse
import cv2 as cv
import numpy as np
from src.simulate_angles import AngleSimulation
from src.Setup import Setup
from src.Trainer import Trainer
from threading import Thread
from typing import Optional


class Demo:
    def __init__(self, input_device: str = "camera", with_angles: bool = False):
        self.setup = Setup(input_device, with_angles=with_angles)

    def run(self):
        self.setup.run(False, None)

    def create_dataset(self, dataset_dir):
        self.setup.run(True, dataset_dir)

    def stop(self):
        self.setup.running = False


classes = ['Pain', 'Neutral']
quitFlag = False


def create_demo(args):
    demo = None
    if args.camera:
        demo = Demo("camera", args.with_angles)
    elif args.realsense:
        demo = Demo("realsense", args.with_angles)
    elif args.video:
        demo = Demo("video", args.with_angles)
    elif args.network:
        demo = Demo("network", args.with_angles)
    return demo


def simulate_angles():
    angle_sim = AngleSimulation()
    cv.namedWindow("TEST INPUT")
    global quitFlag
    while not quitFlag:
        cv.imshow("TEST INPUT", np.zeros([100, 100, 3]))
        key = cv.waitKey(1)
        angle_sim.process_key(key)
        angle_sim.send()


def main():
    # Flags:
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_angles', default=False, type=bool, help='with sending angles')
    parser.add_argument('--network', default=True, type=bool, help='get input from network?')
    parser.add_argument('--camera', default=False, type=bool, help='get input from camera?')
    parser.add_argument('--realsense', default=False, type=bool, help='get input from realsense?')
    parser.add_argument('--video', default=False, type=bool, help='get input from video?')
    parser.add_argument('--train', default=False, type=bool, help='train pain mode?')
    parser.add_argument('--create_dataset', default=False, type=bool, help='create your own dataset?')
    parser.add_argument('--dataset_dir', default="dataset/hdt/realsense", type=str, help='path to the dataset')

    args = parser.parse_args()
    if args.train:
        train_path = [args.dataset_dir + "/" + x + '/*.png' for x in ['Neutral/train', 'Neutral/train_smile', 'Pain/train']]
        test_path = [args.dataset_dir + "/" + x + '/*.png' for x in ['Neutral/train', 'Neutral/train_smile', 'Pain/train']]
        model = Trainer(classes, train_path)
        model.train("svm")  # knn or svm
        model.test(test_path)
        return

    # print(repr(args))
    demo = create_demo(args)  # type: Optional[Demo]
    if demo is None:
        thread_function = lambda: simulate_angles()
    elif args.create_dataset:
        thread_function = lambda: demo.create_dataset(args.dataset_dir)
    else:
        thread_function = lambda: demo.run()
    t = Thread(target=thread_function)
    t.start()

    global quitFlag
    while True:
        x = input()
        if x == "q":
            quitFlag = True
            demo.stop()
            break
        elif x == "r":
            quitFlag = True
            demo.stop()
            t.join()
            demo = create_demo(args)
            quitFlag = False
            t = Thread(target=thread_function)
            t.start()
    t.join()


if __name__ == '__main__':
    main()
