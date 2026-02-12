import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 19999))
s.settimeout(10)
print("UDP listener ready on 0.0.0.0:19999", flush=True)
try:
    data, addr = s.recvfrom(1024)
    print(f"SUCCESS: Got '{data.decode()}' from {addr}", flush=True)
except socket.timeout:
    print("FAIL: No UDP packet received within 10 seconds", flush=True)
s.close()
