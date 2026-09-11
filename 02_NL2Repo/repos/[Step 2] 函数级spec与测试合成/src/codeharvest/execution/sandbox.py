"""Docker 沙箱：启动容器并运行执行服务。"""

import io
import os
import tarfile
from time import sleep

import docker
from docker.models.containers import Container

class DockerSandbox:
    def __init__(
        self,
        image_name: str = "codeharvest_docker_v4",
        repo_id: str = "aider",
        port: int = 3006,
        command: str = "/bin/bash",
        **docker_kwargs,
    ):
        self.image_name = image_name
        self.repo_id = repo_id
        self.command = command
        self.client = docker.from_env()
        self.start_container(image_name, command, port, **docker_kwargs)
        self.workdir = f"/repos/{repo_id}"
        self.start_server(repo_id, port)

    def start_container(
        self, image_name: str, command: str, port: int, **docker_kwargs
    ):
        self.container: Container = self.client.containers.run(  
            image_name,
            command,
            detach=True,
            tty=True,
            ports={f"{port}/tcp": port},
            
            **docker_kwargs,
        )
        try:
            while self.container.status != "running":
                sleep(1)
                self.container.reload()
        except Exception as e:
            print("Container start error", repr(e))
            self.stop_container()

    def run_single_command(self, command: str):
        try:
            exit_code, output = self.container.exec_run(
                command,
                workdir=self.workdir,
                
            )
            if exit_code != 0:
                
                pass
            else:
                
                
                pass
        except Exception as e:
            
            self.stop_container()
        return

    def start_server(self, repo_id: str, port: int):
        command = f"bash -c \
            'source .venv/bin/activate && ls && \
            repo-exec-server start --port {port} &\
            '"

        self.run_single_command(command)
        return

    def stop_container(self):
        try:
            self.container.stop()
            self.container.remove()
        except Exception as e:
            print("Container stop error", repr(e))

if __name__ == "__main__":
    ds = DockerSandbox()
