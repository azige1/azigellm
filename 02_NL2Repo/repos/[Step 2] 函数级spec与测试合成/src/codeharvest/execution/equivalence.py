"""等价性判定：把测试提交给执行服务并解析日志。"""

import json
import rpyc
import random
import traceback

from codeharvest.core import FunctionTestTarget, MethodTestTarget
from codeharvest.execution.service import SandboxServiceManager
from codeharvest.execution.payloads import build_execution_payloads

from codeharvest.logging_utils import exec_logger as logger

def run_target_mp(args) -> tuple[bool, str, FunctionTestTarget | MethodTestTarget]:
    fut: FunctionTestTarget | MethodTestTarget = args[0]
    local: bool = args[1]
    image: str = args[2]
    port = random.randint(3000, 10000)
    
    output = run_target_on_port(fut, port, local, image)
    return output

def run_target_on_port(
    fut: FunctionTestTarget | MethodTestTarget,
    port: int,
    local: bool = False,
    image: str = "codeharvest:temp",
    reuse_port: bool = False,
) -> tuple[bool, str, FunctionTestTarget | MethodTestTarget]:
    try:
        simulator, conn = SandboxServiceManager.get_service(fut.repo_id, port, local, image)
    except Exception as e:
        print("Service error@", fut.repo_id, repr(e))
        fut.test_history.update_exec_stats({"error": repr(e)})
        return False, repr(e), fut

    try:
        return verify_self_equivalence([fut], conn, local)
    except Exception as e:
        tb = traceback.format_exc()
        pass
    finally:
        if simulator:
            simulator.stop_container()
        conn.close()
        if not reuse_port:
            SandboxServiceManager.close_connection(port)

    fut.test_history.update_exec_stats({"error": tb})
    print(f"Error@{fut.repo_id}:\n{tb}")

    return False, tb, fut

def verify_self_equivalence(
    futs: list[FunctionTestTarget | MethodTestTarget],
    conn: rpyc.Connection,
    local: bool = False,
) -> tuple[bool, str, FunctionTestTarget | MethodTestTarget]:
    service = conn.root
    assert service is not None, "Test service is None"

    repo_data, fut_data, test_data = build_execution_payloads(futs, local=local)

    
    service.setup_repo(repo_data)
    service.setup_function(fut_data)
    service.setup_test(test_data)

    init_response = service.init()
    init_output = str(init_response["output"])
    init_error = str(init_response["error"])

    
    
    
    ignore_patterns = ["SyntaxWarning: invalid escape sequence", "NameError: "]

    if init_error and not any(p in init_error for p in ignore_patterns):
        futs[0].test_history.update_exec_stats(
            {"output": init_output, "error": init_error}
        )
        logger.error(f"Init Error@{futs[0].id}:\n{init_error}\n\n")
        return False, init_error, futs[0]

    

    try:
        submit_response = service.submit()
    except Exception as e:
        futs[0].test_history.update_exec_stats({"error": repr(e)})
        logger.error(f"Submit Error@{futs[0].id}:\n{repr(e)}\n\n")
        return False, repr(e), futs[0]

    submit_error = str(submit_response["error"])

    if "logs" not in submit_response:
        futs[0].test_history.update_exec_stats({"error": submit_error})
        logger.error(f"Submit Error@{futs[0].id}:\n{submit_error}\n\n")
        return False, submit_error, futs[0]

    submit_logs = json.loads(submit_response["logs"])
    submit_logs["output"] = submit_response["output"]
    futs[0].test_history.update_exec_stats(submit_logs)

    valids = [x["valid"] for x in submit_logs["run_tests_logs"].values()]
    return all(valids), submit_error, futs[0]

def check_code_equivalence(
    code: str,
    fut: FunctionTestTarget | MethodTestTarget,
    port: int,
    local: bool = False,
    image: str = "codeharvest:temp",
    reuse_port: bool = False,
):

    try:
        simulator, conn = SandboxServiceManager.get_service(fut.repo_id, port, local, image)
    except Exception as e:
        print("Service error@", fut.repo_id, repr(e))
        fut.test_history.update_exec_stats({"error": repr(e)})
        return False, repr(e), fut

    try:
        fut = [fut]
        service = conn.root
        assert service is not None, "Test service is None"

        repo_data, fut_data, test_data = build_execution_payloads(fut, local=local)
        service.setup_repo(repo_data)
        service.setup_function(fut_data)
        service.setup_test(test_data)

        init_response = service.init()
        init_output = str(init_response["output"])
        init_error = str(init_response["error"])

        ignore_patterns = ["SyntaxWarning: invalid escape sequence", "NameError: "]

        if init_error and not any(p in init_error for p in ignore_patterns):
            logger.error(f"Init Error:\n{init_error}\n\n")
            return False, init_error, fut

        
        service.setup_codegen_mode()
        exec_response = service.execute(code)
        exec_output = str(exec_response["output"])
        exec_error = str(exec_response["error"])

        
        try:
            submit_response = service.submit()
        except Exception as e:
            logger.error(f"Submit Error:\n{repr(e)}\n\n")
            return False, repr(e), fut

        submit_error = str(submit_response["error"])

        if "logs" not in submit_response:
            logger.error(f"Submit Error:\n{submit_error}\n\n")
            return False, submit_error, fut

        submit_logs = json.loads(submit_response["logs"])
        submit_logs["output"] = submit_response["output"]
        valids = [x["valid"] for x in submit_logs["run_tests_logs"].values()]
        return all(valids), submit_error, submit_logs["run_tests_logs"]

    except Exception as e:
        tb = traceback.format_exc()
        pass
    finally:
        if simulator:
            simulator.stop_container()
        conn.close()
        if not reuse_port:
            SandboxServiceManager.close_connection(port)

    print(f"Error:\n{tb}")
    return False, tb, {"error": tb}
