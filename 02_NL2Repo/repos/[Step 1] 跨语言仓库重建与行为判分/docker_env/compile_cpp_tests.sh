#!/bin/bash
# 批量编译 cpp2rust 数据集中的 C++ 测试程序（Linux x86_64，容器内编译）。
#
# 对每个库目录下的 test*.cpp，用 g++ -std=c++17 编成与源文件同名（去掉 .cpp）的
# 可执行文件，放在源文件旁——判分器按这个命名约定找参考二进制。
#
# 用法:
#   DATASET_ROOT=/path/to/dataset/cpp2rust \
#   BUILD_IMAGE=<your-registry>/cpp2rust-arena:latest \
#   bash docker_env/compile_cpp_tests.sh
set -e

DATASET_ROOT="${DATASET_ROOT:?请用 DATASET_ROOT 指向 cpp2rust 数据集目录}"
BUILD_IMAGE="${BUILD_IMAGE:?请用 BUILD_IMAGE 指定带 g++ 的镜像}"
CONTAINER_DIR="/workspace/dataset"

compile_one() {
    local repo_name=$1
    local test_file=$2   # 相对库目录的路径，可能是 tests/test1.cpp
    local module_name
    module_name=$(basename "$test_file" .cpp)
    local output_name="${module_name}"

    local host_cpp_path="${DATASET_ROOT}/${repo_name}/${test_file}"
    if [[ ! -f "$host_cpp_path" ]]; then
        echo "[SKIP] 找不到 ${repo_name}/${test_file}"
        return 1
    fi

    echo "[COMPILE] ${repo_name}/${test_file} -> ${output_name}"
    docker run --rm \
        -v "${DATASET_ROOT}:${CONTAINER_DIR}" \
        -w "${CONTAINER_DIR}/${repo_name}" \
        "${BUILD_IMAGE}" \
        bash -c "
        g++ -std=c++17 -O2 -o ${output_name} ${test_file} -lm -pthread 2>&1 || \
        g++ -std=c++17 -O2 -o ${output_name} tests/$(basename "$test_file") -lm -pthread 2>&1 || \
        g++ -std=c++17 -O2 -o tests/${output_name} tests/$(basename "$test_file") -lm -pthread 2>&1
        " && echo "[OK] ${repo_name}/${test_file}" || echo "[FAIL] ${repo_name}/${test_file}"
}

cd "$DATASET_ROOT"

# 不硬编码库清单：数据集目录下的每个子目录都视为一个库
for repo in */; do
    repo=${repo%/}
    echo "===== 编译 $repo ====="
    while IFS= read -r test_file; do
        compile_one "$repo" "${test_file#"$repo"/}"
    done < <(find "$repo" -name "test*.cpp" -type f 2>/dev/null || true)
done

echo "===== 编译完成 ====="
echo "生成的可执行文件（testN 形式、无后缀）："
find "$DATASET_ROOT" -type f -name "test*" ! -name "*.cpp" -perm -u+x 2>/dev/null
