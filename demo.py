import argparse
from src.Setup import Setup
from src.Trainer import Trainer
from threading import Thread
from typing import Optional


class Demo:
    def __init__(self, input_device: str = "camera"):
        self.setup = Setup(input_device)

    def run(self):
        self.setup.run(False, None)

    def create_dataset(self, dataset_dir):
        self.setup.run(True, dataset_dir)

    def stop(self):
        self.setup.running = False


classes = ['Pain', 'Neutral']


def create_demo(args):
    demo = None
    if args.camera:
        demo = Demo("camera")
    elif args.realsense:
        demo = Demo("realsense")
    elif args.video:
        demo = Demo("video")
    elif args.network:
        demo = Demo("network")
    return demo


def main():
    # Flags:
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=True, type=bool, help='get input from network?')
    parser.add_argument('--camera', default=False, type=bool, help='get input from camera?')
    parser.add_argument('--realsense', default=False, type=bool, help='get input from realsense?')
    parser.add_argument('--video', default=False, type=bool, help='get input from video?')
    parser.add_argument('--train', default=False, type=bool, help='train pain mode?')
    parser.add_argument('--create_dataset', default=False, type=bool, help='create your own dataset?')
    parser.add_argument('--dataset_dir', default="./dataset/", type=str, help='path to the dataset')

    args = parser.parse_args()
    if args.train:
        train_path = [args.dataset_dir + c + '/train/*.png' for c in classes]
        test_path = [args.dataset_dir + c + '/test/*.png' for c in classes]
        model = Trainer(classes, train_path)
        model.train("svm")  # knn or svm
        model.test(test_path)

    # print(repr(args))
    demo = create_demo(args)  # type: Optional[Demo]

    if args.create_dataset:
        thread_function = lambda: demo.create_dataset(args.dataset_dir)
    else:
        thread_function = lambda: demo.run()
    t = Thread(target=thread_function)
    t.start()

    while True:
        x = input()
        if x == "q":
            demo.stop()
            break
        elif x == "r":
            demo.stop()
            t.join()
            demo = create_demo(args)
            t = Thread(target=thread_function)
            t.start()
    t.join()


if __name__ == '__main__':
    main()
