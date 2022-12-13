import socket

# UDP variable
# Set IP address as local host, 6100 is destination port
serverAddressPort = ("127.0.0.1", 12000)
bufferSize = 1024
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# message = "I Love Creative Computing!"
message = 123

UDPClientSocket.sendto(str.encode(str(message)), serverAddressPort)
# UDPClientSocket.sendto(str.encode(message), serverAddressPort)