import socket
import struct
from typing import Optional


def send_msg(sock: socket.socket, data: bytes) -> None:
    """
    Send length-prefixed binary message.
    Format:
        4-byte big-endian length + payload
    """
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    Receive exactly n bytes.
    Return None if connection is closed.
    """
    data = b""

    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet

    return data


def recv_msg(sock: socket.socket) -> Optional[bytes]:
    """
    Receive one length-prefixed binary message.
    """
    raw_len = recvall(sock, 4)

    if not raw_len:
        return None

    msg_len = struct.unpack(">I", raw_len)[0]
    return recvall(sock, msg_len)