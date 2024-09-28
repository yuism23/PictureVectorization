#!/bin/bash

# 运行 Evaluation.py 脚本，指定输入和输出路径
CUDA_VISIBLE_DEVICES=2 python Evaluation.py \
    --image_dir /home/hongda/dataset/images \
    --svg_dir /home/hongda/output/svg_files \
    --output_dir /home/hongda/output/visualizations \
    --edge_dir /home/hongda/output/original_edges \
    --svgEdge_folder /home/hongda/output/svg_edges \
    --tolerance 5

# 检查 Evaluation.py 是否成功运行
if [ $? -ne 0 ]; then
    echo "Error: Evaluation.py failed."
    exit 1
fi

echo "Evaluation process completed successfully."
