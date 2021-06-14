import comm
import numpy as np


class AngleSimulation:
    def __init__(self):
        self.data = comm.BytesData(12 * 8)
        self.mode = 0
        self.reset()
        self.comm = comm.Communication()
        self.comm.createSocket(comm.SocketType.UDP, comm.SocketPartner.SocketPartner(("10.162.15.70", 25001), False))
        self.increment = np.pi/12
        self.has_changed = False

    def reset(self):
        for i in range(12):
            self.data.setDouble(0, 8 * i)
        self.mode = 0

    def send(self, verbose: bool = False):
        if verbose and self.has_changed:
            print("Sending data", end=": ")
            for i in range(12):
                print(str(self.data.getDouble(i * 8)) + ", ", end="")
            print()
        self.has_changed = False
        if not self.comm.sendRaw(comm.SocketType.UDP, self.data.getBuffer(), self.data.getBufferSize()):
            print("Could not send simulated angle data to simulink...")
            return False

    def get_mode(self):
        if self.mode == 0:
            return "All angles"
        elif self.mode == 1:
            return "All right hand angles"
        elif self.mode == 2:
            return "right hand angle 1"
        elif self.mode == 3:
            return "right hand angle 2"
        elif self.mode == 4:
            return "right hand angle 3"
        elif self.mode == 5:
            return "right hand angle 4"
        elif self.mode == 6:
            return "right hand angle 5"
        elif self.mode == 7:
            return "right hand angle 6"
        elif self.mode == 8:
            return "All left hand angles"
        elif self.mode == 9:
            return "left hand angle 1"
        elif self.mode == 10:
            return "left hand angle 2"
        elif self.mode == 11:
            return "left hand angle 3"
        elif self.mode == 12:
            return "left hand angle 4"
        elif self.mode == 13:
            return "left hand angle 5"
        elif self.mode == 14:
            return "left hand angle 6"

    def increase(self):
        if self.mode == 0:
            for i in range(12):
                self.data.setDouble(self.data.getDouble(i*8) + self.increment, i*8)
        elif self.mode == 1:
            for i in range(6):
                self.data.setDouble(self.data.getDouble(i * 8) + self.increment, i * 8)
        elif self.mode == 2:
            self.data.setDouble(self.data.getDouble(0) + self.increment, 0)
        elif self.mode == 3:
            self.data.setDouble(self.data.getDouble(8) + self.increment, 8)
        elif self.mode == 4:
            self.data.setDouble(self.data.getDouble(16) + self.increment, 16)
        elif self.mode == 5:
            self.data.setDouble(self.data.getDouble(24) + self.increment, 24)
        elif self.mode == 6:
            self.data.setDouble(self.data.getDouble(32) + self.increment, 32)
        elif self.mode == 7:
            self.data.setDouble(self.data.getDouble(40) + self.increment, 40)
        elif self.mode == 8:
            for i in range(6, 12):
                self.data.setDouble(self.data.getDouble(i * 8) + self.increment, i * 8)
        elif self.mode == 9:
            self.data.setDouble(self.data.getDouble(48) + self.increment, 48)
        elif self.mode == 10:
            self.data.setDouble(self.data.getDouble(56) + self.increment, 56)
        elif self.mode == 11:
            self.data.setDouble(self.data.getDouble(64) + self.increment, 64)
        elif self.mode == 12:
            self.data.setDouble(self.data.getDouble(72) + self.increment, 72)
        elif self.mode == 13:
            self.data.setDouble(self.data.getDouble(80) + self.increment, 80)
        elif self.mode == 14:
            self.data.setDouble(self.data.getDouble(88) + self.increment, 88)

    def decrease(self):
        if self.mode == 0:
            for i in range(12):
                self.data.setDouble(self.data.getDouble(i*8) - self.increment, i*8)
        elif self.mode == 1:
            for i in range(6):
                self.data.setDouble(self.data.getDouble(i * 8) - self.increment, i * 8)
        elif self.mode == 2:
            self.data.setDouble(self.data.getDouble(0) - self.increment, 0)
        elif self.mode == 3:
            self.data.setDouble(self.data.getDouble(8) - self.increment, 8)
        elif self.mode == 4:
            self.data.setDouble(self.data.getDouble(16) - self.increment, 16)
        elif self.mode == 5:
            self.data.setDouble(self.data.getDouble(24) - self.increment, 24)
        elif self.mode == 6:
            self.data.setDouble(self.data.getDouble(32) - self.increment, 32)
        elif self.mode == 7:
            self.data.setDouble(self.data.getDouble(40) - self.increment, 40)
        elif self.mode == 8:
            for i in range(6, 12):
                self.data.setDouble(self.data.getDouble(i * 8) - self.increment, i * 8)
        elif self.mode == 9:
            self.data.setDouble(self.data.getDouble(48) - self.increment, 48)
        elif self.mode == 10:
            self.data.setDouble(self.data.getDouble(56) - self.increment, 56)
        elif self.mode == 11:
            self.data.setDouble(self.data.getDouble(64) - self.increment, 64)
        elif self.mode == 12:
            self.data.setDouble(self.data.getDouble(72) - self.increment, 72)
        elif self.mode == 13:
            self.data.setDouble(self.data.getDouble(80) - self.increment, 80)
        elif self.mode == 14:
            self.data.setDouble(self.data.getDouble(88) - self.increment, 88)

    def process_key(self, key: int):
        if key == ord('d'):
            self.mode = (self.mode + 1) % 15
            print(self.get_mode())
        elif key == ord('a'):
            self.mode = (self.mode + 14) % 15
            print(self.get_mode())
        elif key == ord('w'):
            self.has_changed = True
            self.increase()
        elif key == ord('s'):
            self.has_changed = True
            self.decrease()
