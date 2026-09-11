"""Docker 镜像构建：基础镜像 + 各仓库镜像。"""

import logging
import re
import traceback
import docker
import docker.errors
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rebuild_spec.core.constants import (
    BASE_IMAGE_BUILD_DIR,
    REPO_IMAGE_BUILD_DIR,
)
from rebuild_spec.core.specs import specs_from_records
from rebuild_spec.core.gitops import make_file_logger, release_logger

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class ImageBuildError(Exception):
    """镜像构建失败时抛出，携带镜像名与日志位置。"""

    def __init__(self, image_name: str, message: str, logger: logging.Logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = ""
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def build_one_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
) -> None:
    """在给定的构建上下文目录里构建一个镜像，并全程记录日志。

    参数
    ----
    image_name    : 目标镜像名
    setup_scripts : {脚本名: 脚本内容}，会写进构建上下文
    dockerfile    : Dockerfile 内容
    platform      : 目标平台
    client        : Docker 客户端
    build_dir     : 构建上下文目录（日志与脚本也放在这里）
    nocache       : 是否禁用构建缓存
    """
    logger = make_file_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # 把安装脚本写进构建上下文
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )
        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
        )

        # 持续把构建输出写进日志（去掉 ANSI 颜色码）
        for chunk in response:
            if "stream" in chunk:
                chunk_stream = ansi_escape.sub("", chunk["stream"])
                logger.info(chunk_stream.strip())
        logger.info("Image built successfully!")
    except docker.errors.APIError as e:
        logger.error(f"docker.errors.APIError during {image_name}: {e}")
        raise ImageBuildError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise ImageBuildError(image_name, str(e), logger) from e
    finally:
        release_logger(logger)  # 谁创建谁关闭


def ensure_base_images(
    client: docker.DockerClient, dataset: list, dataset_type: str
) -> None:
    """确保数据集所需的基础镜像已存在，缺失时才构建。

    dataset_type 取值：scratch / swebench / simple。
    """
    test_specs = specs_from_records(dataset, dataset_type, absolute=True)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    for image_name, (dockerfile, platform) in base_images.items():
        try:
            client.images.get(image_name)
            print(f"Base image {image_name} already exists, skipping build.")
            continue
        except docker.errors.ImageNotFound:
            pass
        print(f"Building base image ({image_name})")
        build_one_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def pending_repo_builds(
    client: docker.DockerClient, dataset: list, dataset_type: str
) -> dict[str, Any]:
    """返回仍需构建的仓库镜像：{镜像名: {安装脚本, dockerfile, 平台}}。"""
    image_scripts = dict()
    test_specs = specs_from_records(dataset, dataset_type, absolute=True)

    for test_spec in test_specs:
        # 基础镜像必须先就位
        try:
            client.images.get(test_spec.base_image_key)
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.repo_image_key}\n."
                "Please build the base images first."
            )

        image_exists = False
        try:
            client.images.get(test_spec.repo_image_key)
            image_exists = True
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            image_scripts[test_spec.repo_image_key] = {
                "setup_script": test_spec.setup_script,
                "dockerfile": test_spec.repo_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_all_repo_images(
    client: docker.DockerClient,
    dataset: list,
    dataset_type: str,
    max_workers: int = 4,
    verbose: int = 1,
) -> tuple[list[str], list[str]]:
    """并发构建数据集所需的全部仓库镜像。

    返回 (成功的镜像名列表, 失败的镜像名列表)。
    """
    ensure_base_images(client, dataset, dataset_type)
    configs_to_build = pending_repo_builds(client, dataset, dataset_type)
    if len(configs_to_build) == 0:
        print("No repo images need to be built.")
        return [], []
    print(f"Total repo images to build: {len(configs_to_build)}")

    successful, failed = list(), list()
    with tqdm(
        total=len(configs_to_build), smoothing=0, desc="Building repo images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    build_one_image,
                    image_name,
                    {"setup.sh": config["setup_script"]},
                    config["dockerfile"],
                    config["platform"],
                    client,
                    REPO_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
                ): image_name
                for image_name, config in configs_to_build.items()
            }

            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                    successful.append(futures[future])
                except ImageBuildError as e:
                    print(f"ImageBuildError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    if len(failed) == 0:
        print("All repo images built successfully.")
    else:
        print(f"{len(failed)} repo images failed to build.")

    return successful, failed


__all__ = []
