#!/bin/bash

# 获取当前脚本所在目录
current_dir=$(pwd)

# 遍历当前目录下的所有文件
for file in $current_dir//*.py; do
  # 判断文件是否为 Python 文件
  if [ -f $file ] && [[ $file =~ .*\.py$ ]]; then
    # 运行 Python 文件
    echo $file
    python $file
  fi
done
