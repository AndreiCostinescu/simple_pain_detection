import comm
import cv2 as cv
import dlib
import numpy as np
import pyrealsense2 as rs
from src.Trainer import SVM, preprocess_single_hog, KNearest
from typing import Optional, Tuple


class Setup:
    def __init__(self, input_device: str, model="svm"):
        if model == "svm":
            # using SVM model
            self.svm = SVM()
            self.model_svm = self.svm.load('model_svm.dat')
        else:
            self.knn = KNearest()
            self.model_knn = self.knn.load('model_knn.dat')

        # running
        self.running = True

        # use dlib face landmark
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # number of your own images
        self.num_ur_images = 1000

        self.input_device = input_device
        self.camera = None
        self.image_height = 480
        self.image_width = 640
        self.fps = 30

        self.pain_sender = comm.Communication()
        self.pain_sender.createSocket(comm.SocketType.UDP, comm.SocketPartner.SocketPartner(("127.0.0.1", 25002), False))
        self.pain_sender_data = comm.StatusData(8)  # 8 = sizeof(double)

    def setup_realsense(self):
        try:
            # Set up for realsense
            self.camera = rs.pipeline(rs.context())
            cfg = rs.config()

            cfg.enable_stream(rs.stream.color, self.image_width, self.image_width, rs.format.rgb8, self.fps)

            self.camera.start(cfg)

            # Skip 5 first frames to give the Auto-Exposure time to adjust
            for x in range(5):
                self.camera.wait_for_frames(3000)
            return True
        except Exception as e:
            print(e)
        return False

    def setup_camera(self):
        self.camera = cv.VideoCapture(0)
        return self.camera.isOpened()

    def setup_video(self):
        raise NotImplementedError()

    def setup_network(self):
        print("Initializing network image receiving")
        server = comm.TCPServer(8400)
        while self.running:
            self.camera = server.acceptCommunication()
            if self.camera is not None:
                print("Created partner: " + self.camera.getPartnerString(comm.SocketType.TCP))
                break
        server.cleanup()
        if not self.running:
            print("Demo stopped!")
        return self.running and self.camera is not None

    def setup_device(self):
        if self.input_device == "realsense":
            return self.setup_realsense()
        elif self.input_device == "camera":
            return self.setup_camera()
        elif self.input_device == "video":
            return self.setup_video()
        elif self.input_device == "network":
            return self.setup_network()
        return False

    def cleanup_device(self):
        if self.input_device == "realsense":
            assert isinstance(self.camera, rs.pipeline)
            self.camera.stop()
        elif self.input_device == "camera":
            assert isinstance(self.camera, cv.VideoCapture)
            self.camera.release()
        elif self.input_device == "video":
            assert isinstance(self.camera, cv.VideoCapture)
            self.camera.release()
        elif self.input_device == "network":
            assert isinstance(self.camera, comm.Communication)
            self.camera.cleanup()

    def get_image_from_device(self) -> Optional[np.ndarray]:
        image = None
        if self.input_device == "realsense":
            # Realsense read frame
            frameset = self.camera.wait_for_frames()
            color_frame = frameset.get_color_frame()
            # Verify that frames are valid
            if not color_frame:
                return None
            image = np.asanyarray(color_frame.get_data())
        elif self.input_device == "camera":
            success, image = self.camera.read()
            if not success:
                return None
        elif self.input_device == "video":
            image = None
        elif self.input_device == "network":
            assert isinstance(self.camera, comm.Communication)
            i = comm.ImageData()
            s = comm.StatusData()
            while self.running:
                success, messageType = self.camera.recvMessageType(comm.SocketType.TCP)
                if not success:
                    return None
                if messageType not in [comm.MessageType.IMAGE, comm.MessageType.STATUS]:
                    continue
                if messageType == comm.MessageType.STATUS:
                    data = s
                else:
                    data = i
                success, data = self.camera.recvData(comm.SocketType.TCP,
                                                     data)  # type: bool, (comm.ImageData or comm.StatusData)
                if not success:
                    return None
                if messageType == comm.MessageType.STATUS:
                    assert isinstance(data, comm.StatusData)
                    s = data
                    if comm.strcmp(s.getData(), comm.Messages.QUIT_MESSAGE) == 0:
                        return None
                else:
                    assert isinstance(data, comm.ImageData)
                    i = data
                    if not i.isImageDeserialized():
                        return None
                    image = i.getImage()
                    break
            if not self.running:
                return None

        return image

    def crop_face(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple], Optional[Tuple]]:
        gray = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)

        # using dlib
        faces = self.detector(gray)
        if not faces:
            return False, None, None, None

        # only select the first person
        face = faces[0]
        landmarks = self.predictor(image=gray, box=face)
        x1 = landmarks.part(17).x
        y1 = landmarks.part(19).y
        x2 = landmarks.part(26).x
        y2 = landmarks.part(6).y
        if (y2 - y1 <= 0) or (x2 - x1 <= 0):
            return False, None, None, None

        return True, gray[y1:y2, x1:x2], (x1, y1), (x2, y2)

    def is_pain(self, image: np.ndarray, face: np.ndarray, up_left_corner: Tuple,
                down_right_corner: Tuple) -> Tuple[np.ndarray, bool]:
        img_hog = preprocess_single_hog(face)
        sample_array = np.array([img_hog])
        prediction = self.svm.predict(sample_array)[0]

        pain = (prediction == 0)  # 0: Pain, 1: Neutral
        text = "Pain" if pain else "Neutral"
        color = (1, 1, 255) if pain else (1, 255, 1)

        image = cv.rectangle(image, up_left_corner, down_right_corner, color, 2)
        cv.putText(image, "%s" % text, (up_left_corner[0], up_left_corner[1] - 10), 0, 5e-3 * 100, color, 2)
        return image, pain

    def run(self, create_dataset: bool = False, dataset_dir: str = None):
        # initialization
        if not self.setup_device():
            print("Could not initialize device")
            return
        counter = 0

        try:
            while self.running:
                image = self.get_image_from_device()
                if image is None:
                    print("Input device stopped!")
                    break
                # print(image.shape)
                cv.imshow("Captured image", image)
                key = cv.waitKey(1)
                if key == ord('q') or key == 27:
                    break

                success, cropped_face, up_left_corner, bottom_right_corner = self.crop_face(image)
                if not success:
                    # print("No face detected in image")
                    continue

                resized_face = cv.resize(cropped_face.copy(), (48, 48), cv.INTER_AREA)
                if create_dataset and counter < self.num_ur_images:
                    data_dir = dataset_dir + "/own_data/" + str(counter) + ".png"
                    print("creating your own data ... ")
                    cv.imwrite(data_dir, resized_face)
                    counter += 1
                elif create_dataset and counter >= self.num_ur_images:
                    print("recording is finished")
                    print("please move your own data to the correct train and test path")
                    break
                else:
                    image, pain_flag = self.is_pain(image, resized_face, up_left_corner, bottom_right_corner)
                    cv.imshow("is pain?", image)
                    self.pain_sender_data.setData(comm.doubleToNetworkBytes(b"", 0, pain_flag), 8)
                    if not self.pain_sender.sendData(comm.SocketType.UDP, self.pain_sender_data, True):
                        print("Error when sending data: " + self.pain_sender.getErrorString())

                key = cv.waitKey(1)
                if key == ord('q') or key == 27:
                    break
        except Exception as e:
            print("Exception occurred: " + str(e))

        self.cleanup_device()
        cv.destroyAllWindows()
