#!/bin/bash
# 安装 Apptainer（rootless 容器运行时，SIF 路线需要）
set -e

# 1) 添加 Apptainer PPA 并更新索引
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update

# 2) 安装（默认 rootless）
# sudo apt install -y apptainer
# 若需要 setuid 模式，改装下面这个包：
sudo apt install -y apptainer-suid

sudo mount -o remount,hidepid=0 /proc
