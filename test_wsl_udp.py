"""UDP connectivity test: WSL → Windows, all address combinations."""
import socket, sys, threading, time

PORT = 19877
BIND_ADDRS = ["0.0.0.0", "192.168.0.176", "127.0.0.1"]
SEND_TARGETS = ["192.168.0.176", "127.0.0.1"]

results = {}

def listen_and_report(bind_addr, port, timeout=3):
    """Listen on bind_addr:port and report what we receive."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((bind_addr, port))
        s.settimeout(timeout)
        data, addr = s.recvfrom(1024)
        msg = data.decode()
        results[f"{bind_addr}:{port}"] = f"OK from {addr}: {msg}"
        s.close()
        return True
    except socket.timeout:
        results[f"{bind_addr}:{port}"] = "TIMEOUT"
        s.close()
        return False
    except Exception as e:
        results[f"{bind_addr}:{port}"] = f"ERROR: {e}"
        return False

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "listen"
    
    if mode == "send":
        # Called from WSL: send UDP to all targets
        target_port = int(sys.argv[2])
        for target in SEND_TARGETS:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                msg = f"wsl_to_{target}"
                s.sendto(msg.encode(), (target, target_port))
                print(f"  Sent '{msg}' → {target}:{target_port}")
                s.close()
            except Exception as e:
                print(f"  Send to {target}:{target_port} FAILED: {e}")
        print("  Done sending")
    else:
        # Windows: listen on each bind address
        print(f"\n{'='*60}")
        print(f"UDP Connectivity Test: WSL → Windows")
        print(f"{'='*60}\n")
        
        for bind_addr in BIND_ADDRS:
            for send_target in SEND_TARGETS:
                port = PORT
                PORT += 1
                label = f"WSL→{send_target} (Win bound {bind_addr})"
                print(f"Testing: {label} on port {port}...")
                
                # Start listener in thread
                t = threading.Thread(target=listen_and_report, args=(bind_addr, port))
                t.start()
                time.sleep(0.3)
                
                # Have WSL send to the specific target
                import subprocess
                cmd = f"python3 -c \"import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(b'wsl_to_{send_target}', ('{send_target}', {port})); s.close(); print('sent')\""
                try:
                    r = subprocess.run(["wsl", "bash", "-c", cmd], capture_output=True, text=True, timeout=5)
                    if r.stdout.strip():
                        print(f"  WSL: {r.stdout.strip()}")
                    if r.stderr.strip():
                        print(f"  WSL err: {r.stderr.strip()}")
                except Exception as e:
                    print(f"  WSL command failed: {e}")
                
                t.join(timeout=5)
                key = f"{bind_addr}:{port}"
                status = results.get(key, "UNKNOWN")
                color_ok = status.startswith("OK")
                symbol = "✓" if color_ok else "✗"
                print(f"  {symbol} {label}: {status}\n")
        
        print(f"{'='*60}")
        print("SUMMARY:")
        for key, val in results.items():
            symbol = "✓" if val.startswith("OK") else "✗"
            print(f"  {symbol} {key}: {val}")
        print(f"{'='*60}\n")
