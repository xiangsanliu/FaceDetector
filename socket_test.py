import socket

if __name__ == '__main__':
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.0.107', 4001))
    message = "01,10,00,5A,00,02,04,00,01,01,01,e7,7c"
    print(message.encode())
    s.sendall(message.encode())