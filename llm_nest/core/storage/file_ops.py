from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(target: Path, data: bytes) -> None:
    """原子写入：先写临时文件再 rename，避免写入中断导致文件损坏"""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        os.rename(tmp_path, target)
    except BaseException:
        os.close(fd) if not _fd_closed(fd) else None
        _unlink_safe(Path(tmp_path))
        raise


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """检查磁盘空间是否足够"""
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(target)
    available = stat.f_bavail * stat.f_frsize
    return available >= required_bytes


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """计算文件哈希"""
    import hashlib

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fd_closed(fd: int) -> bool:
    try:
        os.fstat(fd)
        return False
    except OSError:
        return True


def _unlink_safe(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
