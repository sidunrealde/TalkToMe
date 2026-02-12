import aioice.ice
try:
    addrs = aioice.ice.get_host_addresses(use_ipv4=True, use_ipv6=False)
except TypeError:
    addrs = aioice.ice.get_host_addresses()
print("ICE host addresses:", addrs)

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
print("Outbound IP:", s.getsockname()[0])
s.close()
