# cpp2rust 判分/做题容器镜像
#
# 基础镜像不包含在本作业包内：请准备任意带 g++(C++17) 与 python3 的 Linux 镜像，
# 构建后把镜像名通过环境变量 REBUILD_DOCKER_IMAGE 告诉 runner。
#
#   docker build -f docker_env/cpp2rust.Dockerfile -t <your-registry>/cpp2rust-arena:latest .
#   export REBUILD_DOCKER_IMAGE=<your-registry>/cpp2rust-arena:latest
FROM <your-registry>/cpp2rust-base:latest

# 安装 Rust 工具链（发行版自带的 rustc 即可）
RUN apt-get update && apt-get install -y rustc && rm -rf /var/lib/apt/lists/*

# 兼容旧版 glibc 动态链接路径：预编译的 C++ 参考二进制可能按
# /lib64/ld-linux-x86-64.so.2 查找动态链接器，这里补一组软链。
# 若参考二进制链接的 libstdc++ 版本比镜像内的新，请自行把新版
# libstdc++.so.6 放到 /opt/compiler/gcc-12/lib/ 下（本包不含二进制文件）。
RUN mkdir -p /lib64 /opt/compiler/gcc-12/lib /opt/compiler/gcc-12/lib64 \
    && ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2 \
    && ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /opt/compiler/gcc-12/lib64/ld-linux-x86-64.so.2 \
    && cd /opt/compiler/gcc-12/lib \
    && ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 . 2>/dev/null || true \
    && ln -sf /usr/lib/x86_64-linux-gnu/libm.so.6 . 2>/dev/null || true \
    && ln -sf /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 . 2>/dev/null || true \
    && ln -sf /usr/lib/x86_64-linux-gnu/libc.so.6 . 2>/dev/null || true

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]
