import socket


def get_local_ip() -> str:
    # hostname = socket.gethostname()
    # print(f"hostname: {hostname}")
    # ip = socket.gethostbyname_ex(hostname)[2]
    # print(f"IP: {ip}")
    #    ip = "172.20.11.72"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close
    return ip


def main():
    local_ip = get_local_ip()
    print(f"local IP-Adress: {local_ip}")


if __name__ == "__main__":
    main()
