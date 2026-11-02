"""本地 Docker 容器的底层操作：文件进出、生命周期管理、限时执行。"""

from __future__ import annotations

import docker
import logging
import os
import signal
import tarfile
import threading
import time
import traceback
from pathlib import Path
from io import BytesIO
from typing import Optional, List

import docker.errors
from docker.models.containers import Container

# heredoc 结束标记，特意取一个不太可能撞车的值
HEREDOC_DELIMITER = "REBUILD_SPEC_EOF_7391"


def copy_to_container(container: Container, src: Path, dst: Path) -> None:
    """把本地文件拷进容器：先打包成 tar，传输后在容器内解开。"""
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=src.name)

    with open(tar_path, "rb") as tar_file:
        data = tar_file.read()

    container.exec_run(f"mkdir -p {dst.parent}")

    container.put_archive(os.path.dirname(dst), data)
    container.exec_run(f"tar -xf {dst}.tar -C {dst.parent}")

    # 本地与容器内的临时 tar 都清掉
    tar_path.unlink()
    container.exec_run(f"rm {dst}.tar")


def copy_from_container(container: Container, src: Path, dst: Path) -> None:
    """把容器内文件取回本地（带路径穿越防护）。"""
    if not isinstance(src, Path):
        src = Path(src)

    if not isinstance(dst, Path):
        dst = Path(dst)

    if not dst.parent.exists():
        os.makedirs(dst.parent)

    stream, stat = container.get_archive(str(src))

    tar_stream = BytesIO()
    for chunk in stream:
        tar_stream.write(chunk)
    tar_stream.seek(0)

    with tarfile.open(fileobj=tar_stream, mode="r") as tar:

        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)

            prefix = os.path.commonprefix([abs_directory, abs_target])

            return prefix == abs_directory

        def safe_extract(
            tar: tarfile.TarFile,
            path: str = ".",
            members: Optional[List[tarfile.TarInfo]] = None,
            *,
            numeric_owner: bool = False,
        ) -> None:
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")

            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tar, path=str(dst.parent))

    # 解出来的文件名是 src.name，必要时改成目标名
    extracted_file_path = dst.parent / src.name
    if extracted_file_path != dst:
        extracted_file_path.rename(dst)


def write_to_container(container: Container, data: str, dst: Path) -> None:
    """用 heredoc 把一段文本写进容器内的文件。"""
    command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
    container.exec_run(command)


def cleanup_container(
    client: docker.DockerClient,
    container: Container,
    logger: logging.Logger,
) -> None:
    """停止并删除容器；API 停不掉时按 PID 强杀。"""
    if not container:
        return

    container_id = container.id

    try:
        if container:
            logger.info(f"Attempting to stop container {container.name}...")
            container.kill()
    except Exception as e:
        logger.error(
            f"Failed to stop container {container.name}: {e}. Trying to forcefully kill..."
        )
        try:
            assert container_id is not None
            container_info = client.api.inspect_container(container_id)
            pid = container_info["State"].get("Pid", 0)

            if pid > 0:
                logger.info(
                    f"Forcefully killing container {container.name} with PID {pid}..."
                )
                os.kill(pid, signal.SIGKILL)
            else:
                logger.error(
                    f"PID for container {container.name}: {pid} - not killing."
                )
        except Exception as e2:
            raise Exception(
                f"Failed to forcefully kill container {container.name}: {e2}\n"
                f"{traceback.format_exc()}"
            )

    try:
        logger.info(f"Attempting to remove container {container.name}...")
        container.remove(force=True)
        logger.info(f"Container {container.name} removed.")
    except Exception as e:
        raise Exception(
            f"Failed to remove container {container.name}: {e}\n"
            f"{traceback.format_exc()}"
        )


def image_exists_locally(
    client: docker.DockerClient, image_name: str, tag: str, logger: logging.Logger
) -> bool:
    """检查本地是否已有指定 image:tag。"""
    images = client.images.list(name=image_name)
    for image in images:
        if f"{image_name}:{tag}" in image.tags:
            logger.info(f"Using {image_name}:{tag} found locally.")
            return True
    logger.info(f"{image_name}:{tag} cannot be found locally")
    return False


def pull_image_from_registry(
    client: docker.DockerClient, image_name: str, tag: str, logger: logging.Logger
) -> None:
    """从镜像仓库拉取镜像，失败时抛出带说明的异常。"""
    try:
        client.images.pull(image_name, tag=tag)
        logger.info(f"Loaded {image_name}:{tag} from the registry.")
    except docker.errors.ImageNotFound:
        raise Exception(f"Image {image_name}:{tag} not found in the registry.")
    except docker.errors.APIError as e:
        raise Exception(f"Error pulling image: {e}")


def create_container(
    client: docker.DockerClient,
    image_name: str,
    container_name: str,
    logger: logging.Logger,
    user: Optional[str] = None,
    command: Optional[str] = "tail -f /dev/null",
    nano_cpus: Optional[int] = None,
) -> Container:
    """基于指定镜像启动一个常驻容器；镜像不在本地则先拉取。"""
    image, tag = image_name.split(":")
    if not image_exists_locally(client, image, tag, logger):
        pull_image_from_registry(client, image, tag, logger)

    container = None
    try:
        logger.info(f"Creating container for {image_name}...")
        container = client.containers.run(
            image=image_name,
            name=container_name,
            user=user,
            command=command,
            nano_cpus=nano_cpus,
            detach=True,
        )
        logger.info(f"Container for {image_name} created: {container.id}")
        return container
    except Exception as e:
        # 启动失败时把半成品容器清掉再抛错
        logger.error(f"Error creating container for {image_name}: {e}")
        logger.info(traceback.format_exc())
        assert container is not None
        cleanup_container(client, container, logger)
        raise


def exec_run_with_timeout(
    container: Container, cmd: str, timeout: Optional[int] = 60
) -> tuple[str, bool, float]:
    """在容器里执行命令并限制时长，返回 (输出, 是否超时, 耗时秒数)。"""
    exec_result = ""
    exec_id = None
    timed_out = False

    def run_command() -> None:
        nonlocal exec_result, exec_id
        try:
            exec_id = container.client.api.exec_create(container=container.id, cmd=cmd)[  # pyright: ignore
                "Id"
            ]
            exec_stream = container.client.api.exec_start(exec_id=exec_id, stream=True)  # pyright: ignore
            for chunk in exec_stream:
                exec_result += chunk.decode("utf-8", errors="replace")
        except docker.errors.APIError as e:
            raise Exception(f"Container {container.id} cannot execute {cmd}.\n{str(e)}")

    # 单开线程跑命令，主线程负责掐表
    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        if exec_id is not None:
            exec_pid = container.client.api.exec_inspect(exec_id=exec_id)["Pid"]  # pyright: ignore
            container.exec_run(f"kill -TERM {exec_pid}", detach=True)
        timed_out = True
    end_time = time.time()
    return exec_result, timed_out, end_time - start_time


__all__ = []
