"""命令行入口：串起「抓取仓库 → 抽取函数 → 合成测试 → 执行判分」的完整流程。"""

import os
import click
import ast
import json
import textwrap
import subprocess

from codeharvest.extraction import RepoJobConfig, RepoFetcher, extract_repo_callables, render_dockerfile
from codeharvest.extraction.local_setup import setup_all_repo_venvs
from codeharvest.synthesis.tests import TestGenConfig, EquivalenceTestGenerator, GenExecConfig, GenerateExecuteAgent
from codeharvest.synthesis.specs.generator import SpecRefiner
from codeharvest.synthesis.specs.config import SpecGenConfig
from codeharvest.synthesis.specs.solvability_check import SolvabilityChecker, SolvabilityConfig
from codeharvest.synthesis.tasks.export import export_tasks, ExportConfig
from codeharvest.execution.config import ExecutionConfig
from codeharvest.execution.runner import EquivalenceRunner
from codeharvest.execution.service import SandboxServiceManager
from codeharvest.evaluation.test_quality import summarize_results

from codeharvest.utils.io import read_callable_records, read_test_targets
from codeharvest.core import *

from codeharvest.workspace import REPOS_DIR, EXTRACTED_DATA_DIR, TESTGEN_DIR, EXECUTION_DIR

@click.group()
def cli():
    pass

@cli.command()
@click.option('--port', default=3006, type=int, help="执行服务监听端口")
def serve(port):
    """启动 rpyc 执行服务（Docker 沙箱内使用）。"""
    SandboxServiceManager.run_server_forever(port)

def print_result_path(file_path):
    click.secho(f"\nResult: {file_path}", fg="green", bold=True)

@cli.command()
@click.option('--repo_url', '-r', help="待抓取仓库的 Git 地址")
@click.option('--local_repo_path', '-l', help="本地仓库路径")
@click.option('--repo_paths_file', '-p', help="包含若干本地路径的 JSON 清单文件")
@click.option('--repo_urls_file', '-u', help="包含若干 Git 地址的 JSON 清单文件")
@click.option('--cloning_multiprocess', '-m', default=16, type=int, help="抓取仓库时的并行进程数")
@click.option('--run_pycg_for_repos', is_flag=True, help="是否为仓库运行 PyCG 构建调用图")
@click.option('--pycg_timeout', default=5, type=int, help="单个仓库 PyCG 的超时时间（分钟）")
@click.option('--pycg_multiprocess', default=16, type=int, help="运行 PyCG 的并行进程数")
def fetch(**kwargs):
    RepoFetcher.fetch_and_prepare(RepoJobConfig(**kwargs))
    click.echo("仓库准备完成。")
    print_result_path(REPOS_DIR)

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID，用于标识 Docker 镜像")
@click.option('--local', is_flag=True, default=False, help="在本机安装依赖而不构建 Docker 镜像")
@click.option('--install-batch-size', '-k', default=10, type=int, help="容器内并行安装仓库的批大小")
def build(**kwargs):
    repo_count = len([d for d in os.listdir(REPOS_DIR) if os.path.isdir(os.path.join(REPOS_DIR, d))])
    if repo_count == 0:
        click.echo("仓库目录为空，请先运行 fetch 命令。")
        return
    
    click.echo(f"仓库目录中共有 {repo_count} 个仓库。")
    
    if kwargs['local']:
        click.echo("本机模式：为每个仓库创建独立虚拟环境（uv venv）并安装依赖。")
        setup_all_repo_venvs()

    else:
        click.echo("Docker 模式。")
        click.echo("正在生成 Dockerfile...")
        render_dockerfile(RepoJobConfig(**kwargs))
    
        click.secho("警告：构建镜像耗时较长（建议配合 tmux）。是否继续？[y/n]", fg="yellow")        
        if input().lower() != 'y':
            click.echo("已取消。")
            return
        
        click.echo("\n正在构建 Docker 镜像...")
        
        click.secho(f"警告：全部 {repo_count} 个仓库都会被装入容器。是否继续？[y/n]", fg="yellow")
        if input().lower() != 'y':
            click.echo("已取消。")
            return
        
        exp_id = kwargs['exp_id']
        path_to_dockerfile = REPOS_DIR / "codeharvest_final_dockerfile.dockerfile"
        cmd = f"docker build -t codeharvest:{exp_id} -f {path_to_dockerfile} ."
        
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=REPOS_DIR)
        except subprocess.CalledProcessError as e:
            click.secho(f"Docker 镜像构建失败: {e}", fg="red")
            return
    
        click.secho("\nDocker 镜像构建完成。", fg="green")

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID，作为抽取结果文件名的前缀")
@click.option('--overwrite_extracted', '-o', is_flag=True, help="是否覆盖已有的抽取结果")
@click.option('--extraction_multiprocess', '-m', default=16, type=int, help="抽取函数/方法的并行进程数")
@click.option('--disable_dunder_methods', is_flag=True, default=False, help="关闭双下划线方法过滤")
@click.option('--disable_no_docstring', is_flag=True, default=False, help="关闭「无 docstring」过滤")
@click.option('--disable_signature_filters', is_flag=True, default=False, help="关闭函数签名过滤（参数、返回值）")
@click.option('--disable_keyword_filters', is_flag=True, default=False, help="关闭关键词过滤（docstring、函数体、名称）")
@click.option('--disable_wrapper_filters', is_flag=True, default=False, help="关闭包装器过滤（装饰器等）")
@click.option('--disable_lines_filter', is_flag=True, default=False, help="关闭代码行数过滤")
@click.option('--disable_nondeterministic_filter', is_flag=True, default=False, help="关闭「不可对照（不确定行为）」过滤")
@click.option('--disable_all_filters', is_flag=True, default=False, help="关闭全部过滤器")
def harvest(**kwargs):
    repo_args = RepoJobConfig(**kwargs)
    extract_repo_callables(repo_args)
    click.echo("抽取完成。")
    print_result_path(os.path.join(EXTRACTED_DATA_DIR, f"{repo_args.exp_id}_extracted.json"))

def generation_options(f):
    options = [
        click.option('--context_type', default="sliced", help="发给大模型的上下文类型"),
        click.option('--oversample_rounds', default=1, type=int, help="过采样轮数"),
        click.option('--max_context_size', default=32000, type=int, help="上下文最大长度"),
        click.option('--test_style', default="equivalence", type=click.Choice(["equivalence", "literal"]), help="测试写法：equivalence 对照参考实现 / literal 字面期望值"),
        click.option('--disable_captured_calls', is_flag=True, default=False, help="关闭真实调用采集记录的附加"),
        click.option('--capture_exp_id', default=None, help="读取哪个实验 ID 的采集记录，缺省与 exp_id 相同"),
        click.option('--limit', default=None, type=int, help="最多处理多少个函数"),
        click.option('--save_chat', is_flag=True, default=False, help="是否保存对话消息"),
    ]
    for opt in reversed(options):
        f = opt(f)
    return f

def model_options(f):
    options = [
        click.option('--multiprocess', '-m', default=8, type=int, help="并行进程数"),
        click.option('--model_name', default=None, help="使用的大模型名称，缺省用各子命令配置的默认模型"),
        click.option('--base_url', default=None, help="OpenAI 兼容服务地址，缺省读 OPENAI_BASE_URL"),
        click.option('--n', default=None, type=int, help="每次请求生成的补全条数，缺省用子命令默认值"),
        click.option('--top_p', default=0.95, type=float, help="核采样概率 top_p"),
        click.option('--max_tokens', default=None, type=int, help="最大生成 token 数，缺省用子命令默认值"),
        click.option('--temperature', default=None, type=float, help="采样温度，缺省用子命令默认值"),
        click.option('--presence_penalty', default=0.0, type=float, help="presence penalty"),
        click.option('--frequency_penalty', default=0.0, type=float, help="frequency penalty"),
        click.option('--stop', multiple=True, default=[], help="停止序列"),
        click.option('--openai_timeout', default=60, type=int, help="OpenAI 接口超时时间（秒）"),
        click.option('--use_cache', is_flag=True, default=True, help="是否缓存大模型请求结果，默认开启"),
        click.option('--cache_batch_size', default=30, type=int, help="缓存写盘的批大小")
    ]
    for opt in reversed(options):
        f = opt(f)
    return f

def _drop_unset(kwargs: dict) -> dict:
    """去掉未显式传入（值为 None）的选项，让各配置类的默认值生效。"""
    return {k: v for k, v in kwargs.items() if v is not None}

def _fallback_gen_input(ctx, param, value):
    if not value:
        exp_id = ctx.params.get('exp_id')
        default_file = f"{exp_id}_extracted.json"
        click.echo(f"Warning: --in_file not provided. Using `{default_file}` as per `exp_id`.")
        return default_file
    return value

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID，作为生成测试文件名的前缀")
@click.option('--function', '-f', default=None, help="要查看的函数名")
@click.option('--in_file', '-i', callback=_fallback_gen_input, help="测试生成器的输入文件，缺省为 {exp_id}_extracted.json")
@generation_options
@model_options
def synthesize(**kwargs):
    kwargs['use_captured_calls'] = not kwargs.pop('disable_captured_calls')
    test_gen_args = TestGenConfig(**_drop_unset(kwargs))
    EquivalenceTestGenerator.generate(test_gen_args)
    click.echo("测试生成完成。")
    print_result_path(os.path.join(TESTGEN_DIR, f"{test_gen_args.exp_id}_generate.json"))

def execution_options(f):
    options = [
        click.option('--local', is_flag=True, default=False, help="在本机运行执行服务，默认为 Docker"),
        click.option('--image', default="codeharvest:temp", help="运行测试所用的 Docker 镜像名"),
        click.option('--execution-multiprocess', '-m', default=20, type=int, help="执行函数/方法的并行进程数"),
        click.option('--port', default=3006, type=int, help="执行服务端口：串行时默认 3006，并行时随机分配"),
        click.option('--timeout-per-task', default=180, type=int, help="执行服务完成单个任务的超时时间（秒）"),
        click.option('--batch-size', default=100, type=int, help="每执行多少个函数写一次结果文件")
    ]
    for opt in reversed(options):
        f = opt(f)
    return f

def _fallback_exec_input(ctx, param, value):
    if not value:
        exp_id = ctx.params.get('exp_id')
        default_file = f"{exp_id}_generate.json"
        click.echo(f"Warning: --in_file not provided. Using `{default_file}` as per `exp_id`.")
        return default_file
    return value

@cli.command()
@click.option('--exp-id', '-e', default="temp", help="测试执行的实验 ID")
@click.option('--function', '-f', default=None, help="要查看的函数名")
@click.option('--in-file', '-i', callback=_fallback_exec_input, help="测试执行的输入文件")
@execution_options
def run_tests(**kwargs):
    
    if kwargs['local']:
        click.echo("提示：当前在本机运行执行服务，去掉 --local 则使用 Docker。")
        kwargs['image'] = "codeharvest:temp"
    elif not kwargs['image']:
        click.echo(f"Warning: --image not provided. Using `codeharvest:{kwargs['exp_id']}` as per `exp_id`.")
        kwargs['image'] = f"codeharvest:{kwargs['exp_id']}"
    elif kwargs['image'] == "codeharvest:temp":
        click.echo("Warning: Using the default image `codeharvest:temp`. Use --image for custom image.")
    else:
        click.echo(f"Note: Using the provided image: {kwargs['image']}")
    
    testgen_file_path = os.path.join(TESTGEN_DIR, kwargs['in_file'])
    if not os.path.exists(testgen_file_path):
        click.echo(f"\nNo generated tests found for experiment ID: {kwargs['exp_id']}")
        return
        
    args = ExecutionConfig(**kwargs)
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    EquivalenceRunner.run(args)
    click.echo("测试执行完成。")
    print_result_path(os.path.join(EXECUTION_DIR, f"{args.exp_id}_out.json"))

@cli.command()
@click.option('--exp-id', '-e', default="temp", help="测试执行的实验 ID")
@click.option('--function', '-f', default=None, help="要查看的函数名")
@click.option('--in-file', '-i', callback=_fallback_gen_input, help="生成-执行循环的输入文件")
@click.option('--max_rounds', '-k', default=3, type=int, help="生成-执行循环的最大轮数")
@click.option('--min-cov', default=0.8, type=float, help="判定测试有效的最小分支覆盖率")
@click.option('--min-valid', default=0.8, type=float, help="数据集中有效题目的最低比例")
@generation_options
@model_options
@execution_options
def agent_loop(**kwargs):
    kwargs['use_captured_calls'] = not kwargs.pop('disable_captured_calls')

    if kwargs['local']:
        click.echo("提示：当前在本机运行执行服务，去掉 --local 则使用 Docker。")
        kwargs['image'] = "codeharvest:temp"
    elif not kwargs['image']:
        click.echo(f"Warning: --image not provided. Using `codeharvest:{kwargs['exp_id']}` as per `exp_id`.")
        kwargs['image'] = f"codeharvest:{kwargs['exp_id']}"
    elif kwargs['image'] == "codeharvest:temp":
        click.echo("Warning: Using the default image `codeharvest:temp`. Use --image for custom image.")
    else:
        click.echo(f"Note: Using the provided image: {kwargs['image']}")
    
    args = GenExecConfig(**_drop_unset(kwargs))
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    GenerateExecuteAgent.agent_loop(args)
    print_result_path(os.path.join(EXECUTION_DIR, f"{args.exp_id}_out.json"))

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID")
@click.option('--depths', default="1,2,3,5,full", help="扫描的切片深度，逗号分隔，full 表示整仓")
@click.option('--in_file', '-i', default=None, help="抽取结果文件名，缺省 {exp_id}_extracted.json")
def slice_report(exp_id, depths, in_file):
    """切片深度扫描报告：各深度下的无缺失符号占比与上下文长度。"""
    from codeharvest.analysis.slicing.report import SliceReportConfig, run_slice_report

    args = SliceReportConfig(exp_id=exp_id, depths=depths, in_file=in_file)
    run_slice_report(args)

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID")
@click.option('--in_file', '-i', default=None, help="抽取结果文件名，缺省 {exp_id}_extracted.json")
@click.option('--test_timeout', default=240, type=int, help="单仓库 pytest 超时（秒）")
@click.option('--capture_multiprocess', '-m', default=1, type=int, help="并行采集进程数")
@click.option('--skip_examples', is_flag=True, default=False, help="跳过 examples/*.py 脚本采集")
def capture(exp_id, in_file, test_timeout, capture_multiprocess, skip_examples):
    """运行仓库自带测试与 examples，采集被测函数的真实调用。"""
    from codeharvest.analysis.instrument.capture_runner import CaptureConfig, run_capture

    args = CaptureConfig(
        exp_id=exp_id,
        in_file=in_file,
        test_timeout=test_timeout,
        capture_multiprocess=capture_multiprocess,
        run_examples=not skip_examples,
    )
    run_capture(args)

def _fallback_spec_input(ctx, param, value):
    if not value:
        exp_id = ctx.params.get('exp_id')
        default_file = f"{exp_id}_out.json"
        click.echo(f"Warning: --in_file not provided. Using `{default_file}` as per `exp_id`.")
        return default_file
    return value

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID")
@click.option('--in_file', '-i', callback=_fallback_spec_input, help="agent-loop 输出文件，缺省 {exp_id}_out.json")
@click.option('--limit', '-l', default=None, type=int, help="最多处理多少个函数")
@model_options
def refine_specs(exp_id, in_file, limit, **kwargs):
    """以 docstring、签名与已通过测试为材料重写规格。"""
    kwargs['limit'] = limit
    kwargs['exp_id'] = exp_id
    kwargs['in_file'] = in_file
    args = SpecGenConfig(**_drop_unset(kwargs))
    SpecRefiner.generate(args)
    print_result_path(os.path.join(EXECUTION_DIR, f"{args.exp_id}_specgen.json"))

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID")
@click.option('--in_file', '-i', default=None, help="规格任务文件，缺省 {exp_id}_specgen.json")
@click.option('--local', is_flag=True, default=False, help="在本机执行隐藏测试，默认为 Docker")
@model_options
def check_solvability(exp_id, in_file, **kwargs):
    """可解性检查：小模型只读规格实现函数，采样跑隐藏测试。"""
    # 未显式传入的模型、采样数与温度沿用 SolvabilityConfig 默认值（Qwen3-8B、n=4、0.8）
    kwargs['exp_id'] = exp_id
    kwargs['in_file'] = in_file
    args = SolvabilityConfig(**_drop_unset(kwargs))
    SolvabilityChecker.run(args)

@cli.command("export-tasks")
@click.option('--exp_id', '-e', default="temp", help="实验 ID")
@click.option('--in_file', '-i', default=None, help="可解性检查结果，缺省 {exp_id}_solvability.json")
@click.option('--tasks_file', default=None, help="规格任务文件，缺省 {exp_id}_specgen.json")
@click.option('--eval_ratio', default=0.1, type=float, help="按仓库切分的评测集比例")
@click.option('--blocklist', default="data/benchmark_repo_blocklist.txt", help="公开基准仓库黑名单")
def export_tasks_command(exp_id, in_file, tasks_file, eval_ratio, blocklist):
    """把可解的任务打包为训练 / 评测 JSONL（按仓库切分，黑名单过滤）。"""
    args = ExportConfig(
        exp_id=exp_id,
        in_file=in_file,
        tasks_file=tasks_file,
        eval_ratio=eval_ratio,
        blocklist=blocklist,
    )
    export_tasks(args)

def summarize_signature_doc(code, max_width=100):
    try:
        tree = ast.parse(code)
        function_def = tree.body[0]
        args = [arg.arg for arg in function_def.args.args]
        defaults = [ast.unparse(default) for default in function_def.args.defaults]
        padded_defaults = [''] * (len(args) - len(defaults)) + defaults
        signature = ', '.join(f'{arg}={default}' if default else arg for arg, default in zip(args, padded_defaults))
        docstring = ast.get_docstring(function_def)
        if docstring:
            docstring = textwrap.shorten(docstring, width=max_width, placeholder="...")
        else:
            docstring = "No docstring available"  
        return signature, docstring
    except:
        return "Could not parse signature", "Could not parse docstring"

def describe_callable(func):
    try:
        full_name = f"{func.name}" if isinstance(func, FunctionRecord) else f"{func.class_name}.{func.name}"
    except:
        full_name = func.name

    signature, docstring = summarize_signature_doc(func.code)
    ftype = "FunctionRecord" if isinstance(func, FunctionRecord) else "MethodRecord"
    return full_name, signature, docstring, ftype

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID，作为抽取结果文件名的前缀")
@click.option('--limit', '-l', default=10, type=int, help="最多列出多少个函数")
@click.option('--detailed', '-d', is_flag=True, default=False, help="显示每个函数的详细信息")
def list_targets(exp_id, detailed, limit):
    file_path = os.path.join(EXTRACTED_DATA_DIR, f"{exp_id}_extracted.json")
    
    if not os.path.exists(file_path):
        click.echo(f"No extracted functions found for experiment ID: {exp_id}")
        return

    functions = read_callable_records(file_path)
    click.echo(f"Total extracted functions/methods: {len(functions)}")

    for i, func in enumerate(functions[:limit], 1):
        full_name, signature, docstring, ftype = describe_callable(func)
        if detailed:
            click.echo(f"\n{i}. {ftype}: {full_name}")
            click.echo(f"   FileRecord: {func.file.relative_file_path}")
            click.echo(f"   Signature: {func.name}({signature})")
            click.echo(f"   Docstring: {docstring}")
        else:
            click.echo(f"{i}. {full_name} ({func.file.relative_file_path}) [{ftype}]")

    if len(functions) > limit:
        click.echo(f"\n... and {len(functions) - limit} more functions/methods. Use --limit to list more.")

    if not detailed:
        click.echo("\nUse --detailed or -d for more information about each function.")
                    
    

@cli.command()
@click.option('--exp_id', '-e', default="temp", help="实验 ID，作为抽取结果文件名的前缀")
@click.option('--fname', '-f', required=False, help="要查看的函数名")
@click.option('--code', '-c', is_flag=True, help="显示函数代码")
@click.option('--test', '-t', is_flag=True, help="显示为该函数生成的测试")
@click.option('--result', '-r', is_flag=True, help="显示为该函数生成的测试")
@click.option('--show-all', '-a', is_flag=True, help="同时显示代码、测试与执行结果")

@click.option('--chat', is_flag=True, help="显示得到最终测试前的对话消息")
@click.option('--summary', is_flag=True, help="显示测试执行的总体统计")
def show(exp_id, fname, code, test, result, show_all, chat, summary):
    if not summary and not fname:
        click.secho("Error: Please provide the name of the function using --fname.", fg="red")
        return
    
    extracted_file_path = os.path.join(EXTRACTED_DATA_DIR, f"{exp_id}_extracted.json")
    testgen_file_path = os.path.join(TESTGEN_DIR, f"{exp_id}_generate.json")
    executed_file_path = os.path.join(EXECUTION_DIR, f"{exp_id}_out.json")
    
    if summary:
        if not os.path.exists(executed_file_path):
            click.echo(f"\nNo executed tests found for experiment ID: {exp_id}")
            return

        click.echo(summarize_results(exp_id))
        return
    
    if not os.path.exists(extracted_file_path):
        click.echo(f"No extracted functions found for experiment ID: {exp_id}")
        return

    functions = read_callable_records(extracted_file_path)
    target_function = next((func for func in functions if func.name == fname), None)
    if not target_function:
        click.echo(f"No function named '{fname}' found in the extracted data.")
        return
    
    full_name, signature, docstring, ftype = describe_callable(target_function)
    click.echo(f"{ftype}: {full_name}")
    click.echo(f"FileRecord: {target_function.file.file_path}")
    
    if code or show_all:
        click.echo("Code:\n")
        click.echo(textwrap.indent(target_function.code, '    '))
    
    if test or show_all:
        if not (os.path.exists(testgen_file_path) or os.path.exists(executed_file_path)):
            click.echo(f"\nNo generated tests found for experiment ID: {exp_id}")
            return

        if os.path.exists(executed_file_path):
            functions_under_test = read_test_targets(executed_file_path)
        else:
            functions_under_test = read_test_targets(testgen_file_path)
            
        target_fut = next((fut for fut in functions_under_test if fut.name == fname), None)
        
        if not target_fut:
            if os.path.exists(testgen_file_path):
                functions_under_test = read_test_targets(testgen_file_path)
                target_fut = next((fut for fut in functions_under_test if fut.name == fname), None)
            else:
                click.echo(f"\nNo generated test found for function '{fname}'.")
                return

        click.echo("\nGenerated Test:")

        for test_name, test_code in target_fut.test_history.latest_tests.items():
            click.echo(f"\n{test_name}:")
            click.echo(textwrap.indent(test_code, '    '))
        
    if result or show_all:
        if not os.path.exists(executed_file_path):
            click.echo(f"\nNo executed tests found for experiment ID: {exp_id}")
            return

        executed_futs = read_test_targets(executed_file_path)
        target_executed_fut = next((fut for fut in executed_futs if fut.name == fname), None)
        
        if not target_executed_fut:
            click.echo(f"\nNo executed test found for function '{fname}'.")
            return

        click.echo("\nTest Results:")

        for test_name, results in target_executed_fut.exec_stats['run_tests_logs'].items():
            click.echo(f"\n{test_name}:")
            click.echo(json.dumps(results, indent=4))
            click.echo(json.dumps(target_executed_fut.coverage, indent=4))
    
    if chat or show_all:
        if not os.path.exists(executed_file_path):
            click.echo(f"\nNo executed tests found for experiment ID: {exp_id}")
            return

        executed_futs = read_test_targets(executed_file_path)
        target_executed_fut = next((fut for fut in executed_futs if fut.name == fname), None)
        
        if not target_executed_fut:
            click.echo(f"\nNo executed test found for function '{fname}'.")
            return

        click.echo("\nChat Messages:")
        chat_messages = target_executed_fut.test_history.latest_chat_messages
        
        if not chat_messages:
            click.echo("No chat messages found.")
            return
        
        for message in chat_messages:
            click.secho(f"{message['role'].capitalize()}:", fg="green")
            truncated_content = '\n'.join(message['content'].split('\n')[:50])
            click.echo(truncated_content)

if __name__ == '__main__':
    cli()
