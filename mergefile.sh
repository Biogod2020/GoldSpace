#!/usr/bin/env bash
# merged_output.py 为最终合并文件名，可自行修改
output="merged_output_jul_31.txt"
# 如果已有旧的合并文件，先删除
rm -f "$output"

# 从当前目录开始递归查找所有 .py 文件
find . -type f -name "*.py" | sort | while IFS= read -r file; do
  # 打印相对路径作为分隔注释
  echo -e "\n# ===== File: ${file#./} =====\n" >> "$output"
  # 将文件内容追加到合并文件
  cat "$file" >> "$output"
done

echo "已生成合并文件：$output"

