import socket

class Control:

    def __init__(self):
        self.lightsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lightsocket.connect(('192.168.0.107', 4001))
        self.locksocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.locksocket.connect(('192.168.0.108', 4001))

    def greenlight(self):
        message = '0110005A00020400000100777c'
        self.lightsocket.sendall(bytes.fromhex(message))

    def redlight(self):
        message = '0110005A0002040101000026d0'
        self.lightsocket.sendall(bytes.fromhex(message))

    def lockon(self):
        message = '0110004A00010202BBe929'
        self.locksocket.sendall(bytes.fromhex(message))

    def closelight(self):
        message = '0110005A0002040000000076ec'
        self.lightsocket.sendall(bytes.fromhex(message))



if __name__ == '__main__':
    control = Control()
    # control.lockon()
    # control.closelight()