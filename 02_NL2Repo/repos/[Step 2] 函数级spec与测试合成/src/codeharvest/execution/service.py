"""执行服务管理：本地或 Docker 模式的连接与回收。"""

import rpyc
import time
from threading import Thread, Event
from rpyc.utils.server import ThreadPoolServer
from repo_exec_server.server import RepoExecService
from codeharvest.execution.sandbox import DockerSandbox

server_stop_event = Event()

def launch_server_in_thread(port: int):
    server = ThreadPoolServer(
        RepoExecService(), port=port, protocol_config={"allow_reuse_address": True}
    )

    def run_server():
        server.start()
        server_stop_event.wait()
        server.close()
        print(f"Server on port {port} stopped")

    server_thread = Thread(target=run_server)
    server_thread.start()
    return server_thread

def request_server_stop():
    server_stop_event.set()

class SandboxServiceManager:
    running_ports = set()
    server_threads = {}

    @staticmethod
    def get_service(
        repo_id: str, port: int, local: bool = False, image: str = "codeharvest:temp"
    ):
        if local:
            
            if not port in SandboxServiceManager.running_ports:
                t = launch_server_in_thread(port)
                SandboxServiceManager.running_ports.add(port)
                SandboxServiceManager.server_threads[port] = t
                time.sleep(3)

            
            try:
                conn = rpyc.connect("localhost", port)
            except Exception as e:
                print(f"Connection error -- {repr(e)} -- {port}")
                raise e

            return None, conn

        return SandboxServiceManager.get_service_docker(image, repo_id, port)

    @staticmethod
    def get_service_docker(image: str, repo_id: str, port: int):
        
        simulator = DockerSandbox(image_name=image, repo_id=repo_id, port=port)

        
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
        for port in list(SandboxServiceManager.running_ports):
            try:
                conn = rpyc.connect("localhost", port)
                service = conn.root
                service.stop_server()
                conn.close()
            except Exception as e:
                print(f"Error closing connection on port {port}: {e}")

        
        request_server_stop()
        for port, thread in SandboxServiceManager.server_threads.items():
            thread.join(timeout=5)
            if thread.is_alive():
                SandboxServiceManager.force_stop_thread(thread)

        SandboxServiceManager.running_ports.clear()
        SandboxServiceManager.server_threads.clear()
        print("All connections closed and servers stopped")

    @staticmethod
    def close_connection(port):
        try:
            conn = rpyc.connect("localhost", port)
            service = conn.root
            service.stop_server()
            conn.close()
        except Exception as e:
            print(f"Error closing connection on port {port}: {e}")

        if port in SandboxServiceManager.running_ports:
            SandboxServiceManager.running_ports.remove(port)

        if port in SandboxServiceManager.server_threads:
            thread = SandboxServiceManager.server_threads[port]
            thread.join(timeout=5)

            if thread.is_alive():
                SandboxServiceManager.force_stop_thread(thread)

            del SandboxServiceManager.server_threads[port]

    @staticmethod
    def force_stop_thread(thread):
        if hasattr(thread, "_tstate_lock"):
            if thread._tstate_lock.locked():
                thread._tstate_lock.release()
        thread._stop()
