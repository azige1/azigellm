"""执行服务管理：本地直连或 Docker 模式的连接与回收。"""

import time
from threading import Thread, Event

from codeharvest.execution.local_exec_service import RepoExecService
from codeharvest.execution.sandbox import DockerSandbox


class DirectConnection:
    """本地直连连接对象：root 为服务实例本身，close 时清理沙箱。

    与 rpyc 连接保持同一套调用界面（conn.root.xxx(...) / conn.close()），
    使 equivalence.py 中的调用方无需区分本地与 Docker 模式。
    """

    def __init__(self, service: RepoExecService):
        self.root = service

    def close(self):
        # 本地模式每个被测函数一个服务实例，关闭时清理其临时沙箱目录
        self.root.stop_server()


class SandboxServiceManager:
    running_ports = set()
    server_threads = {}

    @staticmethod
    def launch_server_in_thread(port: int):
        """在线程中启动 rpyc 服务（供 `codeharvest serve` 与 Docker 模式使用）。"""
        import rpyc
        from rpyc.utils.server import ThreadPoolServer

        server = ThreadPoolServer(
            RepoExecService(), port=port, protocol_config={"allow_reuse_address": True}
        )

        def run_server():
            server.start()

        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        SandboxServiceManager.running_ports.add(port)
        SandboxServiceManager.server_threads[port] = server_thread
        time.sleep(3)
        return server_thread

    @staticmethod
    def run_server_forever(port: int):
        """阻塞式启动 rpyc 服务，供 CLI `codeharvest serve --port` 使用。"""
        import rpyc
        from rpyc.utils.server import ThreadPoolServer

        server = ThreadPoolServer(
            RepoExecService(), port=port, protocol_config={"allow_reuse_address": True}
        )
        print(f"执行服务已启动: localhost:{port}")
        try:
            server.start()
        except KeyboardInterrupt:
            server.close()

    @staticmethod
    def get_service(
        repo_id: str, port: int, local: bool = False, image: str = "codeharvest:temp"
    ):
        if local:
            # 本地直连：不起 rpyc 线程服务，直接持有服务实例
            return None, DirectConnection(RepoExecService())

        return SandboxServiceManager.get_service_docker(image, repo_id, port)

    @staticmethod
    def get_service_docker(image: str, repo_id: str, port: int):
        simulator = DockerSandbox(image_name=image, repo_id=repo_id, port=port)

        import rpyc

        try:
            conn = rpyc.connect(
                "localhost", port, keepalive=True, config={"sync_request_timeout": 180}
            )
        except Exception as e:
            print(f"Connection error -- {repo_id} -- {repr(e)}")
            simulator.stop_container()
            raise e
        return simulator, conn

    @staticmethod
    def shutdown():
        """回收本地遗留的 rpyc 线程服务（本地直连模式无可回收资源）。"""
        for port in list(SandboxServiceManager.running_ports):
            try:
                SandboxServiceManager.close_connection(port)
            except Exception as e:
                print(f"Error closing connection on port {port}: {e}")

        SandboxServiceManager.running_ports.clear()
        SandboxServiceManager.server_threads.clear()
        print("All connections closed and servers stopped")

    @staticmethod
    def close_connection(port):
        """关闭指定端口的 rpyc 线程服务。"""
        if port not in SandboxServiceManager.running_ports:
            return

        thread = SandboxServiceManager.server_threads.get(port)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                SandboxServiceManager.force_stop_thread(thread)
            del SandboxServiceManager.server_threads[port]

        SandboxServiceManager.running_ports.remove(port)

    @staticmethod
    def force_stop_thread(thread):
        """线程超时未退出时仅记录日志，不调用 CPython 私有 API。"""
        print(f"警告: 端口服务线程 {thread.name} 未能及时退出，保留为守护线程")
