"""用 LLM 把任务目录里的 Apptainer .def 转成 Dockerfile，构建镜像并验证初始测试。

流程：读 .def → LLM 转换 → 构建 Docker 镜像 → 容器内跑初始测试。
带 OOM 防护（构建/运行均限制内存）与失败重试入口。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any

from tqdm import tqdm
from openai import OpenAI

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# 允许从仓库根目录运行
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

_openai_client = None
_anthropic_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Run: pip install anthropic")
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

SYSTEM_MSG = """You are an expert in converting Apptainer or Singularity definition files to Dockerfiles.
Convert the given Apptainer .def file to an equivalent Dockerfile.

Key conversion rules:
- Apptainer's `Bootstrap: docker` with `From: ubuntu:22.04` becomes `FROM ubuntu:22.04`
- `%post` section commands become `RUN` commands in Dockerfile
- `%environment` section becomes `ENV` commands
- `%help` section can be added as comments
- Ensure all commands are properly formatted for Docker
- Use `RUN` for each command or chain them with `&&`
- Make sure to install pytest if it's needed
- Preserve user creation and permissions setup
- The home path should be /home/user
- Set WORKDIR to /home/user so commands run from there by default

Heredoc and multiline command rules (IMPORTANT):
- Every shell command must be inside a `RUN` instruction. Do NOT leave lines like `PermitRootLogin yes` or similar outside of a `RUN`; otherwise Docker will treat them as invalid instructions.
- When converting heredoc blocks (e.g. `cat <<EOF ... EOF`):
  - Keep the entire heredoc inside a single `RUN` instruction.
  - The line with `<<EOF` (or `<< 'EOF'`) must be the last thing on that line (no trailing `&&` etc.).
  - The heredoc body must start on the next line and be left as plain text.
  - The terminating `EOF` must be alone on its own line with nothing after it.
  - If needed, you may place any follow-up commands (e.g. `chmod ...`) in a separate `RUN` instruction after the heredoc.
- Do NOT wrap a heredoc block inside a single-quoted string like `sh -c 'cat <<EOF ... EOF'` in a way that causes the heredoc contents to be parsed as Dockerfile instructions.
- When adding comments, use the `#` character and place it at the beginning of the line.
- Always use ubuntu:22.04 from docker. Do not mention .sif containers.

Output only the Dockerfile content, no explanations or markdown code blocks."""


USER_TEMPLATE = """Convert this Apptainer definition file to a Dockerfile:

{def_content}

Output the complete Dockerfile only."""


def read_def_file(def_path: Path) -> Optional[str]:
    """读取 Apptainer 定义文件文本。"""
    return def_path.read_text(encoding="utf-8")


def _extract_dockerfile_content(response_content: str) -> str:
    """去掉模型输出里的 Markdown 围栏，留下纯 Dockerfile 文本。"""
    dockerfile_content = response_content.strip()

    if "```dockerfile" in dockerfile_content or "```Dockerfile" in dockerfile_content:
        parts = dockerfile_content.split("```")
        for i, part in enumerate(parts):
            if "dockerfile" in part.lower() and i + 1 < len(parts):
                dockerfile_content = parts[i + 1].strip()
                break
    elif dockerfile_content.startswith("```"):
        lines = dockerfile_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        dockerfile_content = "\n".join(lines)

    return dockerfile_content.strip()


def _def_to_dockerfile_openai(
    def_content: str,
    model: str = "gpt-5.1",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> Optional[str]:
    """用 OpenAI 模型把 .def 转成 Dockerfile，限流时指数退避。"""
    prompt = USER_TEMPLATE.format(def_content=def_content)
    client = get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response_content = response.choices[0].message.content
            return _extract_dockerfile_content(response_content)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "overloaded" in error_str

            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited (OpenAI), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Error converting to Dockerfile with OpenAI: {e}")
                return None

    return None


def _def_to_dockerfile_anthropic(
    def_content: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> Optional[str]:
    """用 Anthropic 模型把 .def 转成 Dockerfile，限流时指数退避。"""
    prompt = USER_TEMPLATE.format(def_content=def_content)
    client = get_anthropic_client()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_MSG,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            response_content = response.content[0].text
            return _extract_dockerfile_content(response_content)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "overloaded" in error_str

            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited (Anthropic), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Error converting to Dockerfile with Anthropic: {e}")
                return None

    return None


def def_to_dockerfile(
    def_content: str,
    model: str = "gpt-5.1",
    provider: str = "openai",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Optional[str]:
    """.def → Dockerfile 的统一入口。

    参数:
        def_content: Apptainer 定义文件文本。
        model: 模型名。
        provider: ``openai`` 或 ``anthropic``。

    返回:
        Dockerfile 文本；失败返回 None。
    """
    if provider == "anthropic":
        return _def_to_dockerfile_anthropic(
            def_content, model=model, temperature=temperature, max_tokens=max_tokens
        )
    else:
        return _def_to_dockerfile_openai(
            def_content, model=model, temperature=temperature, max_tokens=max_tokens
        )


def pre_pull_base_images(dockerfiles: List[Path]) -> None:
    """预拉取所有 Dockerfile 引用的基础镜像，避免并行构建时重复拉取。"""
    base_images = set()
    for dockerfile_path in dockerfiles:
        if dockerfile_path.exists():
            content = dockerfile_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line.upper().startswith("FROM "):
                    # 兼容 "FROM image AS stage" 写法
                    parts = line.split()
                    if len(parts) >= 2:
                        image = parts[1]
                        base_images.add(image)

    if base_images:
        print(f"Pre-pulling {len(base_images)} base images: {base_images}")
        for image in base_images:
            try:
                subprocess.run(
                    ["docker", "pull", image],
                    timeout=300,
                    capture_output=True,
                )
                print(f"  [ok] Pulled {image}")
            except Exception as e:
                print(f"  [warn] Failed to pull {image}: {e}")


def create_dockerignore(context_dir: Path) -> None:
    """写一个最小 .dockerignore，减少构建上下文传输量。"""
    dockerignore_path = context_dir / ".dockerignore"
    if not dockerignore_path.exists():
        dockerignore_path.write_text(
            "*.sif\n"
            "*.tar\n"
            "*.tar.gz\n"
            "*.zip\n"
            "solutions/\n"
            "__pycache__/\n"
            "*.pyc\n"
            ".git/\n"
        )


def build_image(
    dockerfile_path: Path,
    image_name: str,
    context_dir: Path,
    memory_limit: str = "8g",
    network_mode: str = "host",
) -> bool:
    """带内存上限地构建 Docker 镜像，输出流式打印并检测 OOM 迹象。"""
    create_dockerignore(context_dir)

    # --memory 防 OOM；--network=host 下载更快；--progress=plain 降低输出开销
    cmd = [
        "docker", "build",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,
        "--network", network_mode,
        "--progress=plain",
        "-t", image_name,
        "-f", str(dockerfile_path),
        str(context_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = []
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode('utf-8', errors='replace').rstrip()
            print(decoded)
            output_lines.append(decoded)
            if any(oom_indicator in decoded.lower() for oom_indicator in [
                'out of memory', 'oom', 'cannot allocate memory', 'killed',
                'memory allocation failed', 'enomem'
            ]):
                print(f"[warn] Possible OOM detected during build for {image_name}")

        returncode = proc.wait(timeout=1800)
        if returncode == 0:
            return True
        elif returncode == 137:  # SIGKILL，通常是 OOM
            print(f"Docker build killed (exit 137, likely OOM) for {image_name}")
            return False
        else:
            print(f"Docker build failed with return code {returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Docker build timed out for {image_name}, killing process...")
        if proc is not None:
            proc.kill()
            proc.wait()
        return False
    except FileNotFoundError as e:
        print(f"Docker build error: {e}")
        return False


def run_initial_tests_in_docker(
    image_name: str,
    test_file_path: Path,
    memory_limit: str = "4g",
) -> tuple[bool, str]:
    """在容器里执行初始状态 pytest，返回 ``(是否通过, 输出)``。"""
    task_dir = test_file_path.parent
    test_filename = test_file_path.name

    cmd = [
        "docker", "run", "--rm",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,
        "--oom-kill-disable=false",
        "-v", f"{task_dir}:/mnt",
        image_name,
        "bash", "-c",
        f"cp /mnt/{test_filename} /home/user/ && pytest -v /home/user/{test_filename}",
    ]
    print(f"Running: docker run --rm --memory {memory_limit} -v {task_dir}:/mnt {image_name} bash -c '...'")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = []
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode('utf-8', errors='replace').rstrip()
            print(decoded)
            output_lines.append(decoded)

        returncode = proc.wait(timeout=300)
        output = '\n'.join(output_lines)

        if returncode == 0:
            return True, output
        elif returncode == 137:
            print(f"Test container killed (exit 137, likely OOM) for {image_name}")
            return False, f"Error running tests: OOM killed (exit 137)\n{output}"
        else:
            return False, output
    except subprocess.TimeoutExpired:
        print(f"Test execution timed out for {image_name}, killing process...")
        if proc is not None:
            proc.kill()
            proc.wait()
        return False, "Error running tests: timeout"
    except FileNotFoundError as e:
        return False, f"Error running tests: {e}"


def get_failed_tasks_from_results(result_pattern: str = "retry*.json") -> List[Path]:
    """从当前目录的 result_*.json 中提取失败的任务目录列表。"""
    current_dir = Path.cwd()
    result_files = glob.glob(str(current_dir / result_pattern))

    failed_task_dirs = set()

    for result_file in result_files:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for result in data["results"]:
            if not result.get("success", True):
                task_dir_str = result.get("task_dir")
                task_dir = Path(task_dir_str)
                failed_task_dirs.add(task_dir)

    return list(sorted(list(failed_task_dirs)))


def convert_task_dir(
    task_dir: Path,
    model: str = "gpt-5.1",
    provider: str = "openai",
    skip_build: bool = False,
    skip_tests: bool = False,
    reuse_dockerfile: bool = False,
    reuse_image: bool = False,
    build_memory_limit: str = "8g",
    test_memory_limit: str = "4g",
    network_mode: str = "host",
) -> Dict[str, Any]:
    """处理单个任务目录：.def 转 Dockerfile → 构建 → 跑初始测试。"""
    result = {
        "task_dir": str(task_dir),
        "success": False,
        "error": None,
        "dockerfile_generated": False,
        "docker_build_success": False,
        "tests_passed": False,
    }

    def_path = task_dir / "container.def"
    initial_test_path = task_dir / "test_initial_state.py"
    dockerfile_path = task_dir / "Dockerfile"

    if reuse_dockerfile and dockerfile_path.exists():
        print(f"Reusing existing Dockerfile for {task_dir.name}...")
        dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
        result["dockerfile_generated"] = True
    else:
        def_content = read_def_file(def_path)

        print(f"Converting {task_dir.name} to Dockerfile using {provider}/{model}...")
        dockerfile_content = def_to_dockerfile(
            def_content,
            model=model,
            provider=provider,
        )

        if not dockerfile_content:
            result["error"] = "Failed to generate Dockerfile"
            return result

        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
        result["dockerfile_generated"] = True

    if skip_build:
        result["success"] = True
        return result

    image_name = f"task-img-{task_dir.name.lower().replace('_', '-')}"

    build_success = False
    if reuse_image:
        check_result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
        )
        if check_result.returncode == 0:
            print(f"Reusing existing Docker image {image_name}...")
            result["docker_build_success"] = True
            build_success = True
        else:
            print(f"Building Docker image {image_name}...")
            build_success = build_image(
                dockerfile_path, image_name, task_dir,
                memory_limit=build_memory_limit,
                network_mode=network_mode,
            )
            result["docker_build_success"] = build_success
    else:
        print(f"Building Docker image {image_name}...")
        build_success = build_image(
            dockerfile_path, image_name, task_dir,
            memory_limit=build_memory_limit,
            network_mode=network_mode,
        )
        result["docker_build_success"] = build_success

    if not build_success:
        result["error"] = "Docker build failed"
        return result

    if skip_tests:
        result["success"] = True
        return result

    print(f"Running initial tests for {task_dir.name}...")
    tests_passed, test_output = run_initial_tests_in_docker(
        image_name, initial_test_path,
        memory_limit=test_memory_limit,
    )
    result["tests_passed"] = tests_passed

    if tests_passed:
        result["success"] = True
    else:
        result["error"] = f"Tests failed: {test_output[:500]}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SIF files to Dockerfiles and verify initial tests pass"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("docker_conversion_results.json"),
        help="File to save the results to",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("tasks"),
        help="Directory containing task directories",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model to use for conversion (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use (openai or anthropic)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker build (only generate Dockerfile)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (only build Docker image)",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Start at task number",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of tasks to process",
    )
    parser.add_argument(
        "--reuse-dockerfile",
        action="store_true",
        help="Reuse existing Dockerfile if present (skip LLM conversion)",
    )
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        help="Reuse existing Docker image if present (skip build)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers to process tasks in parallel",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Read result_*.json files in current directory and retry only failed tasks",
    )
    parser.add_argument(
        "--pre-pull",
        action="store_true",
        help="Pre-pull base images before building (avoids redundant pulls during parallel builds)",
    )
    parser.add_argument(
        "--build-memory",
        type=str,
        default="8g",
        help="Memory limit for Docker build (e.g., 8g, 16g). Prevents OOM crashes.",
    )
    parser.add_argument(
        "--test-memory",
        type=str,
        default="4g",
        help="Memory limit for Docker test runs (e.g., 4g, 8g). Prevents OOM crashes.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="host",
        choices=["host", "bridge", "none"],
        help="Docker network mode for builds. 'host' is fastest for package downloads.",
    )
    args = parser.parse_args()


    if args.retry_failed:
        task_dirs = get_failed_tasks_from_results()
        print(f"Found {len(task_dirs)} failed task directories from result files")
        task_dirs = task_dirs[args.start_at:args.start_at + args.num_tasks]
    else:
        # 只处理已有 o3 汇总且 pass@16 > 0 的任务
        task_dirs = [Path(args.task_dir) / f for f in os.listdir(args.task_dir) if "task" in f]
        task_dirs = [f for f in task_dirs if (f / "solutions" / "o3_summary.json").exists()]
        task_dirs = [f for f in task_dirs if json.load(open(f / "solutions" / "o3_summary.json"))["pass_at_k"]["16"] > 0]
        task_dirs = list(sorted(task_dirs))
        task_dirs = task_dirs[args.start_at:args.start_at + args.num_tasks]
        print(f"Found {len(task_dirs)} task directories")

    if args.pre_pull and not args.skip_build:
        dockerfiles = [task_dir / "Dockerfile" for task_dir in task_dirs]
        pre_pull_base_images(dockerfiles)

    results = []

    if args.workers <= 1:
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            result = convert_task_dir(
                task_dir,
                model=args.model,
                provider=args.provider,
                skip_build=args.skip_build,
                skip_tests=args.skip_tests,
                reuse_dockerfile=args.reuse_dockerfile,
                reuse_image=args.reuse_image,
                build_memory_limit=args.build_memory,
                test_memory_limit=args.test_memory,
                network_mode=args.network,
            )
            results.append(result)

            if result["success"]:
                print(f"[ok] {task_dir.name}: Success")
            else:
                print(f"[fail] {task_dir.name}: {result.get('error', 'Unknown error')}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    convert_task_dir,
                    task_dir,
                    model=args.model,
                    provider=args.provider,
                    skip_build=args.skip_build,
                    skip_tests=args.skip_tests,
                    reuse_dockerfile=args.reuse_dockerfile,
                    reuse_image=args.reuse_image,
                    build_memory_limit=args.build_memory,
                    test_memory_limit=args.test_memory,
                    network_mode=args.network,
                ): task_dir
                for task_dir in task_dirs
            }

            with tqdm(total=len(task_dirs), desc="Processing tasks") as pbar:
                for future in as_completed(futures):
                    task_dir = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result["success"]:
                            print(f"[ok] {task_dir.name}: Success")
                        else:
                            print(f"[fail] {task_dir.name}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"[fail] {task_dir.name}: Exception - {e}")
                        results.append({
                            "task_dir": str(task_dir),
                            "success": False,
                            "error": f"Exception: {e}",
                            "dockerfile_generated": False,
                            "docker_build_success": False,
                            "tests_passed": False,
                        })
                    finally:
                        pbar.update(1)

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    dockerfile_generated = sum(1 for r in results if r.get("dockerfile_generated", False))
    docker_build_success = sum(1 for r in results if r.get("docker_build_success", False))
    tests_passed = sum(1 for r in results if r.get("tests_passed", False))

    errors_dockerfile = sum(1 for r in results if not r.get("dockerfile_generated", False))
    errors_build = sum(1 for r in results if r.get("dockerfile_generated", False) and not r.get("docker_build_success", False) and not args.skip_build)
    errors_tests = sum(1 for r in results if r.get("docker_build_success", False) and not r.get("tests_passed", False) and not args.skip_tests)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")

    print(f"\nStage Breakdown:")
    print(f"  Dockerfile generated: {dockerfile_generated}/{total} ({dockerfile_generated/total*100:.1f}%)")
    if not args.skip_build:
        print(f"  Docker build succeeded: {docker_build_success}/{total} ({docker_build_success/total*100:.1f}%)")
    if not args.skip_tests:
        print(f"  Tests passed: {tests_passed}/{total} ({tests_passed/total*100:.1f}%)")

    if failed > 0:
        print(f"\nErrors by Stage:")
        if errors_dockerfile > 0:
            print(f"  Dockerfile generation: {errors_dockerfile}")
        if errors_build > 0:
            print(f"  Docker build: {errors_build}")
        if errors_tests > 0:
            print(f"  Test execution: {errors_tests}")

    print("="*60)

    output_file = Path(args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "successful": successful,
                "failed": failed,
                "dockerfile_generated": dockerfile_generated,
                "docker_build_success": docker_build_success,
                "tests_passed": tests_passed,
                "errors_dockerfile": errors_dockerfile,
                "errors_build": errors_build,
                "errors_tests": errors_tests,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
