"""Test UDP connectivity between Windows and WSL"""
import socket
import subprocess
import time
import threading

def udp_send_delayed():
    """Send UDP after a short delay to give listener time to start."""
    time.sleep(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    targets = ["192.168.0.176", "127.0.0.1", "localhost"]
    for target in targets:
        try:
            s.sendto(b"ping", (target, 19999))
            print(f"  Sent UDP to {target}:19999")
        except Exception as e:
            print(f"  Failed to send to {target}: {e}")
    s.close()

def udp_listen():
    """Listen for UDP on Windows side."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 19998))
    s.settimeout(8)
    print("  Windows UDP listener on 0.0.0.0:19998")
    try:
        data, addr = s.recvfrom(1024)
        print(f"  WIN RECEIVED: '{data.decode()}' from {addr}")
    except socket.timeout:
        print(f"  WIN: No UDP received (timeout)")
    s.close()

if __name__ == "__main__":
    # Test 1: Can Windows reach WSL via UDP?
    print("\n=== Test 1: Windows -> WSL (UDP) ===")
    print("  Starting WSL listener via subprocess...")
    wsl_proc = subprocess.Popen(
        ["wsl", "bash", "-c", 
         "python3 -c \""
         "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
         "s.bind(('0.0.0.0', 19999)); s.settimeout(8); "
         "print('WSL listening on 19999', flush=True); "
         "data,addr=s.recvfrom(1024); "
         "print(f'WSL RECEIVED: {data} from {addr}', flush=True); "
         "s.close()\""],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    time.sleep(3)
    
    # Send from Windows
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for target in ["192.168.0.176", "127.0.0.1"]:
        try:
            s.sendto(b"hello_from_windows", (target, 19999))
            print(f"  Sent to {target}:19999")
        except Exception as e:
            print(f"  Send to {target} failed: {e}")
    s.close()
    
    wsl_proc.wait(timeout=12)
    output = wsl_proc.stdout.read()
    print(f"  WSL output: {output.strip()}")
    
    # Test 2: Can WSL reach Windows via UDP?
    print("\n=== Test 2: WSL -> Windows (UDP) ===")
    listener = threading.Thread(target=udp_listen)
    listener.start()
    time.sleep(1)
    
    wsl_proc2 = subprocess.Popen(
        ["wsl", "bash", "-c",
         "python3 -c \""
         "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
         "s.sendto(b'hello_from_wsl', ('192.168.0.17', 19998)); "
         "print('WSL sent to 192.168.0.17:19998', flush=True); "
         "s.close()\""],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    wsl_proc2.wait(timeout=10)
    print(f"  WSL sender: {wsl_proc2.stdout.read().strip()}")
    listener.join(timeout=10)
    
    # Test 3: Can WSL reach Google STUN?
    print("\n=== Test 3: WSL -> STUN server (UDP) ===")
    wsl_proc3 = subprocess.Popen(
        ["wsl", "bash", "-c",
         "python3 -c \""
         "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
         "s.settimeout(5); "
         "s.sendto(bytes([0,1,0,0]+[0]*16), ('stun.l.google.com', 19302)); "
         "print('Sent STUN request', flush=True); "
         "data,addr=s.recvfrom(1024); "
         "print(f'STUN response from {addr}, {len(data)} bytes', flush=True); "
         "s.close()\""],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    wsl_proc3.wait(timeout=10)
    print(f"  {wsl_proc3.stdout.read().strip()}")
    
    print("\n=== Summary ===")
    print("If Test 1 FAILED: Windows Firewall is blocking UDP to WSL (root cause of ICE failure)")
    print("If Test 2 FAILED: Windows Firewall is blocking inbound UDP from WSL")
    print("If Test 3 FAILED: WSL can't reach external STUN server")
