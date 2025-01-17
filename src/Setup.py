import comm
import cv2 as cv
import dlib
import numpy as np
import pyrealsense2 as rs
import traceback
from comm import BytesData, Communication, ImageData, MessageType, StatusData, SocketType, SocketPartner, TCPServer
from src.Trainer import SVM, preprocess_single_hog, KNearest
from src.simulate_angles import AngleSimulation
from typing import List, Optional, Tuple


class Setup:
    def __init__(self, input_device: str, model="svm", with_angles: bool = False):
        if model == "svm":
            # using SVM model
            self.svm = SVM()
            # self.model_svm = self.svm.load('model_svm.dat')
            self.model_svm = self.svm.load('models/model_svm_2021-04-26-19-18-15.dat')
            # self.model_svm = self.svm.load('models/model_svm_2021-04-26-21-17-30.dat')
        else:
            self.knn = KNearest()
            self.model_knn = self.knn.load('models/model_knn.dat')

        # running
        self.running = True

        # use dlib face landmark
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        # number of your own images
        self.num_ur_images = 10000

        self.input_device = input_device
        self.camera = None
        self.image_height = 480
        self.image_width = 640
        self.fps = 30

        # self.receiver_ip = "10.151.11.202"
        self.receiver_ip = "10.162.15.70"
        self.pain_sender = Communication()
        self.pain_sender.createSocket(SocketType.UDP, SocketPartner.SocketPartner((self.receiver_ip, 25002), False))
        self.pain_sender_data = BytesData(8)  # 8 = sizeof(double)

        self.angle_sim = AngleSimulation() if with_angles else None

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
        server = TCPServer(8400)
        while self.running:
            self.camera = server.acceptCommunication()
            if self.camera is not None:
                print("Created partner: " + self.camera.getPartnerString(SocketType.TCP))
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
            assert isinstance(self.camera, Communication)
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
            assert isinstance(self.camera, Communication)
            i = ImageData()
            s = StatusData()
            while self.running:
                success, messageType = self.camera.recvMessageType(SocketType.TCP, False)
                if not success:
                    return None
                if messageType not in [MessageType.IMAGE, MessageType.STATUS]:
                    continue
                if messageType == MessageType.STATUS:
                    data = s
                else:
                    data = i
                success, data = self.camera.recvData(SocketType.TCP, data, False)
                if not success:
                    return None
                if messageType == MessageType.STATUS:
                    assert isinstance(data, StatusData)
                    s = data
                    if comm.strcmp(s.getData(), comm.Messages.QUIT_MESSAGE) == 0:
                        return None
                else:
                    assert isinstance(data, ImageData)
                    i = data
                    if not i.isImageDeserialized():
                        return None
                    image = i.getImage()
                    break
            if not self.running:
                return None

        return image

    def crop_face(self, image: np.ndarray) -> Tuple[
        bool, Optional[np.ndarray], Optional[Tuple], Optional[Tuple], Optional[List]]:
        gray = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)  # type: np.ndarray

        # using dlib
        faces = self.detector(gray)
        if not faces:
            return False, None, None, None, None

        # only select the first person
        face = faces[0]
        landmarks = self.predictor(image=gray, box=face)
        # print(landmarks.parts())
        """
        x1 = landmarks.part(17).x
        y1 = landmarks.part(19).y
        x2 = landmarks.part(26).x
        y2 = landmarks.part(7).y
        """
        points = landmarks.parts()
        x1 = max(0, min([x.x for x in points]))
        x2 = max([x.x for x in points])
        y1 = max(0, min([x.y for x in points]))
        y2 = max([x.y for x in points])
        # """
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or (y2 - y1 <= 0) or (x2 - x1 <= 0):
            return False, None, None, None, []
        face = gray[y1:y2, x1:x2]  # type: np.ndarray
        assert (face.size > 0), "{} // {}: {}, {}, {}, {}".format(face.size, face.shape, x1, y1, x2, y2)
        return True, face, (x1, y1), (x2, y2), points

    def is_pain(self, image: np.ndarray, face: np.ndarray, up_left_corner: Tuple,
                down_right_corner: Tuple, face_features: List) -> Tuple[np.ndarray, bool]:
        img_hog = preprocess_single_hog(face)
        sample_array = np.array([img_hog])
        prediction = self.svm.predict(sample_array)[0]

        pain = (prediction == 0)  # 0: Pain, 1: Neutral
        text = "Pain" if pain else "Neutral"
        color = (1, 1, 255) if pain else (1, 255, 1)
        blue = (255, 1, 1)

        image = cv.rectangle(image, up_left_corner, down_right_corner, color, 2)
        for feature in face_features:
            cv.circle(image, (feature.x, feature.y), 1, blue)
        cv.putText(image, "%s" % text, (up_left_corner[0], up_left_corner[1] - 10), 0, 5e-3 * 100, color, 2)
        return image, pain

    def run(self, create_dataset: bool = False, dataset_dir: str = None):
        # initialization
        if not self.setup_device():
            print("Could not initialize device")
            return
        counter = 0

        cv.namedWindow("Captured image")
        cv.namedWindow("is pain?")
        # print("Entering try block...")
        try:
            while self.running:
                if self.angle_sim is not None:
                    self.angle_sim.send(True)

                # print("running...")
                image = self.get_image_from_device()
                if image is None:
                    print("Input device stopped!")
                    break
                # print(image.shape)
                cv.imshow("Captured image", image)
                # print("after cv.imshow")
                key = cv.waitKey(1)
                # print("k =", key)
                if self.angle_sim is not None:
                    self.angle_sim.process_key(key)
                if key == ord('q') or key == 27:
                    break
                # print("Shown image")

                success, cropped_face, up_left_corner, bottom_right_corner, features = self.crop_face(image)
                if not success:
                    # print("No face detected in image")
                    continue
                if cropped_face.size <= 0:
                    print("Found face with shape:", cropped_face.shape, " and size:", cropped_face.size)
                    continue

                # print("Resize face")
                resized_face = cv.resize(cropped_face, (48, 48), cv.INTER_AREA)
                if create_dataset and counter < self.num_ur_images:
                    data_dir = dataset_dir + "/" + str(counter) + ".png"
                    print("creating your own data at " + data_dir + "... ")
                    if not cv.imwrite(data_dir, resized_face):
                        raise Exception("Could not write to file...")
                    counter += 1
                elif create_dataset and counter >= self.num_ur_images:
                    print("recording is finished")
                    print("please move your own data to the correct train and test path")
                    break
                else:
                    image, pain_flag = self.is_pain(image, resized_face, up_left_corner, bottom_right_corner, features)
                    cv.imshow("is pain?", image)
                    self.pain_sender_data.setDouble(pain_flag, 0)
                    if not self.pain_sender.sendRaw(comm.SocketType.UDP, self.pain_sender_data.getBuffer(),
                                                    self.pain_sender_data.getBufferSize()):
                        print("Error when sending data: " + self.pain_sender.getErrorString())

                key = cv.waitKey(1)
                if self.angle_sim is not None:
                    self.angle_sim.process_key(key)
                if key == ord('q') or key == 27:
                    break
        except Exception as e:
            print("Exception occurred: " + str(e))
            traceback.print_exc()

        self.cleanup_device()
        # cv.destroyAllWindows()
        cv.destroyWindow("Captured image")
        cv.destroyWindow("is pain?")
