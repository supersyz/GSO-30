#!/bin/bash
#conda activate hunyuan3dv2
set -x  # 启用调试模式，显示执行的命令

# ===== 用户配置区域 =====
# 评估目录
EVAL_DIR=/home/syz/project/AIGC3D-Integration/sotas/hunyuan3dv2/hunyuan3-d-2_20250410/eval_cd
# 真值目录
INPUT_DIR=original
# 评估环境
# =====================

echo "Starting ground truth processing pipeline..."

# 切换到评估目录
cd $EVAL_DIR
echo "Working directory: $PWD"

# 1. 对齐GT模型
echo "Aligning ground truth models..."
python align_gt.py --input $INPUT_DIR 

# 2. 转换为GLB格式
echo "Converting OBJ to GLB format..."
python obj_to_glb.py 

# 3. 渲染30个选定的对象
echo "Rendering 30 picked objects..."
python render_gt.py

echo "Ground truth processing completed!" 