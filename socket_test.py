import socket

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.0.107', 4001))
    message = "0110005A00020400010101e77c"
    message = "0110005A00020400000100777c"
    s.sendall(bytes.fromhex(message))