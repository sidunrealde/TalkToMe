import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(b"ping", ("192.168.0.176", 19999))
print("Sent UDP ping to 192.168.0.176:19999")
s.close()
