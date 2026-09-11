"""批量调度：任务收集、单题执行（本地/容器/自修复）、进度与 token 统计。

run_benchmark.py 只负责解析命令行，真正的流程都在这里：
- collect_jobs：扫描数据集目录，挑出还没做过的源文件；
- run_one：单个源文件的完整做题流程（多进程 worker，必须可序列化）；
- run_batch：进程池调度 + 进度落盘 + 用量汇总。
"""

import json
import os
import posixpath
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import config, sandbox
from .llm import run_text_agent, run_tool_agent
from .tasks import get_task_spec
from .tasks.base import (
    container_dataset_path,
    container_tag,
    entry_name,
    library_of,
    package_dir,
    strip_ext,
)


# ---------- 任务收集 ----------

def collect_jobs(task_name, mode, dataset_root, run_root, model_name, self_repair=0):
    """扫描数据集目录，返回待处理的 job 字典列表。

    断点续跑规则：产物目录里已存在入口文件（或已有成功记录）的源文件直接跳过。
    """
    spec = get_task_spec(task_name)
    jobs = []
    for root, _dirs, files in os.walk(dataset_root):
        for file in sorted(files):
            if not file.endswith(spec.source_ext):
                continue
            source_path = os.path.join(root, file)
            rel_path = os.path.relpath(source_path, dataset_root)

            pkg_dir = package_dir(run_root, rel_path, spec.source_ext)
            entry_path = os.path.join(pkg_dir, entry_name(rel_path, spec.source_ext, spec.target_ext))
            if os.path.exists(entry_path):
                print(f"[SKIP] 产物已存在: {rel_path}")
                continue
            if is_already_succeeded(run_root, library_of(rel_path), rel_path):
                print(f"[SKIP] 已成功过: {rel_path}")
                continue

            jobs.append({
                "task_name": task_name,
                "mode": mode,
                "source_path": source_path,
                "rel_path": rel_path,
                "run_root": str(run_root),
                "model_name": model_name,
                "self_repair": self_repair,
            })
    return jobs


# ---------- 成功记录与轨迹日志 ----------

def _success_record_path(run_root, library):
    return os.path.join(str(run_root), "packages", library, "success_records.jsonl")


def is_already_succeeded(run_root, library, rel_path):
    """查成功记录，用于断点续跑。"""
    record_path = _success_record_path(run_root, library)
    if not os.path.exists(record_path):
        return False
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and json.loads(line).get("file_name") == rel_path:
                    return True
    except Exception:
        pass
    return False


def append_success_record(run_root, library, rel_path, pkg_dir, messages):
    """写入成功记录，并把完整对话轨迹存到产物目录。"""
    record_path = _success_record_path(run_root, library)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)

    trajectory_path = os.path.join(pkg_dir, "trajectory.jsonl")
    if messages:
        with open(trajectory_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    entry = json.dumps({
        "file_name": rel_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trajectory": trajectory_path,
    }, ensure_ascii=False)
    with open(record_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def append_trajectory_log(log_dir, run_name, rel_path, messages, result):
    """往 logs/<task>/<run_name>.jsonl 追加一条轨迹。"""
    if not messages:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.jsonl")
    entry = {
        "file_name": rel_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "messages": messages,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------- 提示词上下文 ----------

def build_local_ctx(spec, job):
    """本地模式的提示词上下文：路径全部是宿主机路径。"""
    source_path = os.path.abspath(job["source_path"])
    pkg_dir = package_dir(job["run_root"], job["rel_path"], spec.source_ext)
    library = library_of(job["rel_path"])
    hints = getattr(spec, "forbidden_hints", None)
    return {
        "source_path": source_path,
        "source_code": Path(source_path).read_text(encoding="utf-8"),
        "entry_name": entry_name(job["rel_path"], spec.source_ext, spec.target_ext),
        "out_dir": pkg_dir,
        "work_dir": job["run_root"],
        "forbidden_hint": _forbidden_hint(spec, library),
        "reference_path": None,
    }


def build_container_ctx(spec, job):
    """容器模式的提示词上下文：路径全部是容器内路径（不创建容器）。"""
    source_path = os.path.abspath(job["source_path"])
    library = library_of(job["rel_path"])
    module = os.path.splitext(os.path.basename(job["rel_path"]))[0]
    return {
        "source_path": container_dataset_path(job["rel_path"], spec.source_ext),
        "source_code": Path(source_path).read_text(encoding="utf-8"),
        "entry_name": entry_name(job["rel_path"], spec.source_ext, spec.target_ext),
        "out_dir": config.CONTAINER_OUTPUT_DIR,
        "work_dir": config.CONTAINER_WORKSPACE,
        "forbidden_hint": _forbidden_hint(spec, library),
        "reference_path": posixpath.join(config.CONTAINER_DATASET_DIR, f"{module}_executable"),
    }


def _forbidden_hint(spec, library):
    """取该库的「禁止参照实现」提示。"""
    return spec.forbidden_hints.get(library, "none")


def sample_prompt(job):
    """生成某个 job 的示例提示词（用于开跑前人工确认）。"""
    spec = get_task_spec(job["task_name"])
    ctx = build_container_ctx(spec, job) if job["mode"] == "docker" else build_local_ctx(spec, job)
    return spec.build_prompt(ctx)


# ---------- 单题执行 ----------

def _run_local(job):
    """本地模式：agent 直接在宿主机写产物。"""
    spec = get_task_spec(job["task_name"])
    pkg_dir = package_dir(job["run_root"], job["rel_path"], spec.source_ext)
    os.makedirs(pkg_dir, exist_ok=True)

    ctx = build_local_ctx(spec, job)
    prompt = spec.build_prompt(ctx)

    execute = lambda cmd: sandbox.run_local_shell(cmd, spec.blocked_patterns, spec.block_msg)

    after_command = None
    if spec.name == "cpp2rust":
        cargo_toml = os.path.join(pkg_dir, "Cargo.toml")
        after_command = lambda: sandbox.strip_cargo_dependencies(cargo_toml)

    token_info = run_tool_agent(
        prompt, job["model_name"], execute, after_command=after_command, return_messages=True
    )
    return token_info, pkg_dir


def _run_docker(job):
    """容器模式：起断网容器、拷入源文件与参考二进制、agent 在容器内做题、拷回产物。"""
    spec = get_task_spec(job["task_name"])
    source_path = os.path.abspath(job["source_path"])
    pkg_dir = package_dir(job["run_root"], job["rel_path"], spec.source_ext)
    os.makedirs(pkg_dir, exist_ok=True)

    ref_host, _ref_name = spec.find_reference(source_path)
    if not ref_host:
        return {"error": f"找不到参考可执行文件: {source_path}"}, pkg_dir, None

    ctx = build_container_ctx(spec, job)
    prompt = spec.build_prompt(ctx)

    module = os.path.splitext(os.path.basename(job["rel_path"]))[0]
    tag = container_tag(job["rel_path"], spec.source_ext)
    image = config.docker_image(spec.name)
    container = sandbox.Container.start(f"{config.CONTAINER_PREFIX}-{tag}", image)
    if not container:
        return {"error": "容器创建失败"}, pkg_dir, None

    try:
        container.is_alive()
        # 数据集目录 + 参考二进制（统一改名为 <模块名>_executable）+ 源文件
        sandbox.docker_run(
            f"docker exec {container.name} mkdir -p {config.CONTAINER_DATASET_DIR}", timeout=10
        )
        ref_dst = posixpath.join(config.CONTAINER_DATASET_DIR, f"{module}_executable")
        ok, _, _ = container.copy_in(ref_host, ref_dst, timeout=10)
        if not ok:
            return {"error": "参考可执行文件拷贝失败"}, pkg_dir, container
        sandbox.docker_run(f"docker exec {container.name} chmod +x {ref_dst}", timeout=10)
        src_dst = posixpath.join(
            config.CONTAINER_DATASET_DIR, os.path.basename(job["rel_path"])
        )
        container.copy_in(source_path, src_dst, timeout=10)

        execute = lambda cmd: container.exec(cmd, spec.blocked_patterns, spec.block_msg)
        token_info = run_text_agent(prompt, job["model_name"], execute, return_messages=True)

        # 无论成败都把 /output 拷回宿主机
        ok, _, stderr = container.copy_out(f"{config.CONTAINER_OUTPUT_DIR}/.", pkg_dir, timeout=30)
        if ok:
            print(f"[OK] 产物已拷回: {pkg_dir}")
        else:
            print(f"[WARN] 产物拷回失败: {stderr}")
        return token_info, pkg_dir, container
    finally:
        if container:
            print(f"[CLEANUP] 销毁容器: {container.name}")
            container.remove()


def _validate_equivalence(source_path, entry_path, arg_sets):
    """自修复模式的行为校验：同一组参数分别跑源实现与生成实现，比对 stdout 与退出码。"""
    fail_reason = ""
    for args in arg_sets:
        src = _quick_run(f"{config.PYTHON_BIN} {source_path} {args}")
        if src["exit_code"] != 0:
            print(f"   [Skip] 参数 '{args}' 在源实现上就跑不过，忽略")
            continue
        gen = _quick_run(f"{config.NODE_BIN} {entry_path} {args}")
        if src["stdout"] != gen["stdout"]:
            return False, (
                f"Args '{args}' output mismatch.\n"
                f"Source stdout:\n{src['stdout']}\nGenerated stdout:\n{gen['stdout']}"
            )
        if src["exit_code"] != gen["exit_code"]:
            return False, f"Args '{args}' exit code mismatch ({src['exit_code']} vs {gen['exit_code']})"
    return True, fail_reason


def _quick_run(cmd):
    """30 秒超时的本地快速执行，返回 stdout/stderr/exit_code 字典。"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "exit_code": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": 1}


def _generate_arg_sets(source_code, cases_path, model_name):
    """让 agent 读源码并产出若干组 CLI 参数，存成 JSON 数组。"""
    gen_prompt = f"""
You are a test engineer. Read the following source code and devise 5 different sets of command-line arguments that exercise its logic.
Cover: normal input, boundary values, and plausible error inputs.

Source code:
{source_code}

Output a JSON array of strings only, e.g.: ["--input 1", "--input 10 --verbose", ""]
No explanations, no output values — only argument strings.
"""
    print("生成自校验参数...")
    run_tool_agent(
        f"Read the code and save the test arguments to {cases_path} as a JSON array:\n{gen_prompt}",
        model_name,
        execute=lambda cmd: sandbox.run_local_shell(cmd),
    )
    try:
        with open(cases_path, "r") as f:
            return json.load(f)
    except Exception:
        return [""]


def _run_local_with_repair(job):
    """本地自修复模式（py2node）：生成参数 → 做题 → 行为校验 → 带反馈重试。"""
    spec = get_task_spec(job["task_name"])
    max_attempts = job["self_repair"]
    source_path = os.path.abspath(job["source_path"])
    pkg_dir = package_dir(job["run_root"], job["rel_path"], spec.source_ext)
    entry_path = os.path.join(pkg_dir, entry_name(job["rel_path"], spec.source_ext, spec.target_ext))
    cases_path = os.path.join(job["run_root"], "selfcheck_cases", strip_ext(job["rel_path"], spec.source_ext) + ".json")
    os.makedirs(pkg_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cases_path), exist_ok=True)

    source_code = Path(source_path).read_text(encoding="utf-8")
    ctx = build_local_ctx(spec, job)
    base_prompt = spec.build_prompt(ctx)

    arg_sets = _generate_arg_sets(source_code, cases_path, job["model_name"])
    fail_reason = ""
    execute = lambda cmd: sandbox.run_local_shell(cmd, spec.blocked_patterns, spec.block_msg)

    total = {"input_tokens": 0, "output_tokens": 0, "messages": []}
    for attempt in range(max_attempts):
        print(f"[PID {os.getpid()}] 第 {attempt + 1}/{max_attempts} 次尝试...")
        if attempt == 0:
            prompt = base_prompt
        else:
            prompt = (
                f"{base_prompt}\n\n"
                f"[IMPORTANT] Do not rewrite from scratch — fix the existing code.\n"
                f"Entry file: {entry_path}\n"
                f"Library directory: {pkg_dir}\n"
                f"Previous failure:\n{fail_reason}"
            )
        info = run_tool_agent(prompt, job["model_name"], execute, return_messages=True)
        if info:
            total["input_tokens"] += info.get("input_tokens", 0)
            total["output_tokens"] += info.get("output_tokens", 0)
            total["messages"] = info.get("messages") or total["messages"]

        passed, fail_reason = _validate_equivalence(source_path, entry_path, arg_sets)
        if passed:
            print(f"[PID {os.getpid()}] 自校验通过: {job['rel_path']}")
            return total, pkg_dir
        print(f"[PID {os.getpid()}] 自校验失败: {fail_reason}")

    print(f"[PID {os.getpid()}] {max_attempts} 次尝试后仍未通过: {job['rel_path']}")
    total["error"] = "self-repair attempts exhausted"
    return total, pkg_dir


def run_one(job):
    """单个 job 的完整流程入口（多进程 worker）。

    返回统一的结果字典：success / rel_path / input_tokens / output_tokens / error?。
    """
    rel_path = job["rel_path"]
    run_root = job["run_root"]
    spec = get_task_spec(job["task_name"])
    library = library_of(rel_path)
    container = None
    try:
        if job["mode"] == "docker":
            token_info, pkg_dir, container = _run_docker(job)
        elif job.get("self_repair"):
            token_info, pkg_dir = _run_local_with_repair(job)
        else:
            token_info, pkg_dir = _run_local(job)

        token_info = token_info or {}
        input_tokens = token_info.get("input_tokens", 0)
        output_tokens = token_info.get("output_tokens", 0)
        messages = token_info.get("messages")

        entry_path = os.path.join(
            pkg_dir, entry_name(rel_path, spec.source_ext, spec.target_ext)
        )
        success = (
            not token_info.get("error")
            and os.path.isfile(entry_path)
            and os.path.getsize(entry_path) > 0
        )

        log_dir = os.path.join(str(config.LOG_DIR), spec.name)
        if success:
            append_success_record(run_root, library, rel_path, pkg_dir, messages)
            append_trajectory_log(log_dir, job["model_name"], rel_path, messages, "success")
        else:
            append_trajectory_log(log_dir, job["model_name"], rel_path, messages, "failed")

        result = {
            "success": success,
            "rel_path": rel_path,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_dir": pkg_dir,
        }
        if not success:
            result["error"] = token_info.get("error", "入口产物未生成")
        return result

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"success": False, "rel_path": rel_path, "error": str(exc)}


# ---------- 进程池调度 ----------

def save_progress(progress_file, total, results):
    """进度落盘，中断后可人工检查。"""
    with open(progress_file, "w") as f:
        json.dump({"total": total, "processed": len(results), "results": results}, f, indent=2)


def run_batch(jobs, num_processes, run_root):
    """进程池跑全部 job，打印并落盘 token 用量汇总。"""
    progress_file = os.path.join(str(run_root), "progress.json")
    start_time = time.time()
    results = []

    print(f"\n[START] {num_processes} 个并发进程开始做题...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_job = {executor.submit(run_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERROR] worker 进程崩溃: {job['rel_path']} - {exc}")
                result = {"success": False, "rel_path": job["rel_path"], "error": str(exc)}
            results.append(result)
            print(f"\n[PROGRESS] [{i}/{len(jobs)}] 完成: {result['rel_path']}")
            if i % 5 == 0:
                save_progress(progress_file, len(jobs), results)

    save_progress(progress_file, len(jobs), results)
    _print_summary(results, time.time() - start_time, run_root)
    return results


def _print_summary(results, elapsed, run_root):
    """用量汇总：控制台打印 + token_usage.log 落盘。"""
    ok = [r for r in results if r.get("success")]
    bad = [r for r in results if not r.get("success")]
    total_input = sum(r.get("input_tokens", 0) for r in ok)
    total_output = sum(r.get("output_tokens", 0) for r in ok)

    for r in bad:
        print(f"失败: {r['rel_path']}，原因: {r.get('error', '未知')}")

    lines = [
        "=" * 60,
        "Token 用量统计",
        "=" * 60,
        f"成功: {len(ok)}  失败: {len(bad)}",
        f"总耗时: {elapsed:.2f}s ({elapsed / 60:.2f}min)",
    ]
    if results:
        lines.append(f"平均每题: {elapsed / len(results):.2f}s")
    lines += [
        f"总输入 Token:  {total_input:,}",
        f"总输出 Token:  {total_output:,}",
        f"总 Token:      {total_input + total_output:,}",
    ]
    if ok:
        lines.append(f"平均每题输入:  {total_input / len(ok):,.0f}")
        lines.append(f"平均每题输出:  {total_output / len(ok):,.0f}")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print("\n" + text)

    log_path = os.path.join(str(run_root), "token_usage.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{text}\n")
    print(f"用量日志已写入: {log_path}")
