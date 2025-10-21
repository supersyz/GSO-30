import os
import numpy as np
import trimesh
import logging
import argparse
from typing import Union, Tuple

# 设置日志配置
logging.basicConfig(
    filename='mesh_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    对 Trimesh 对象的顶点进行归一化处理，将模型缩放到边界框 [-0.5, 0.5] 内，并将中心移到原点。
    """
    # 提取原始顶点
    original_vertices = mesh.vertices.copy()
    
    # 找到每个坐标轴的最小值和最大值
    min_coord = original_vertices.min(axis=0)
    max_coord = original_vertices.max(axis=0)
    
    # 计算中心点
    center = (min_coord + max_coord) / 2.0
    
    # 将所有顶点平移到以中心为原点
    vertices_centered = original_vertices - center
    
    # 计算最大的边长
    max_length = (max_coord - min_coord).max()
    
    # 进行等比例缩放，使得模型适应边界框 [-0.5, 0.5]
    vertices_normalized = vertices_centered / max_length
    
    # 更新 Trimesh 对象的顶点
    mesh.vertices = vertices_normalized
    
    return mesh

def rotate_z_to_y_positive(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    将模型从朝向 Z 轴旋转为朝向 Y 正轴。
    绕 X 轴旋转 -90 度，使头部方向从 Z 轴变为 Y 轴。
    """
    rotation_matrix_x = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1]
    ])
    
    mesh.apply_transform(rotation_matrix_x)
    return mesh

def process_single_mesh(input_path: str, output_path: str) -> bool:
    """
    处理单个mesh文件
    
    Args:
        input_path: 输入mesh文件路径
        output_path: 输出mesh文件路径
    
    Returns:
        bool: 处理是否成功
    """
    try:
        if os.path.exists(output_path):
            print(f'文件已存在：{output_path}')
            return True
            
        # 加载网格
        mesh = trimesh.load(input_path)
        
        # 旋转网格
        rotated_mesh = rotate_z_to_y_positive(mesh)
        
        # 归一化旋转后的网格
        normalized_mesh = normalize_mesh(rotated_mesh)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存处理后的网格
        normalized_mesh.export(output_path)
        
        logging.info(f'已处理并保存文件：{output_path}')
        print(f'已处理并保存文件：{output_path}')
        return True
        
    except Exception as e:
        logging.error(f'处理文件 {input_path} 时出错: {e}')
        print(f'处理文件 {input_path} 时出错: {e}')
        return False

def process_directory(input_dir: str, output_dir: str) -> Tuple[int, int]:
    """
    处理整个目录下的mesh文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    
    Returns:
        Tuple[int, int]: (成功处理的文件数, 总文件数)
    """
    success_count = 0
    total_count = 0
    
    # 遍历输入目录下的所有子文件夹
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        
        # 确认是目录
        if not os.path.isdir(subdir_path):
            continue
            
        # 构建输入输出路径
        obj_filename = f'{subdir}.obj'
        obj_filepath = os.path.join(subdir_path, obj_filename)
        
        if not os.path.isfile(obj_filepath):
            logging.warning(f'未找到 .obj 文件：{obj_filepath}')
            print(f'未找到 .obj 文件：{obj_filepath}')
            continue
            
        total_count += 1
        output_subdir = os.path.join(output_dir, subdir)
        output_filename = f'{subdir}_align.obj'
        output_filepath = os.path.join(output_subdir, output_filename)
        
        if process_single_mesh(obj_filepath, output_filepath):
            success_count += 1
            
    return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description='处理3D mesh文件：旋转和归一化')
    parser.add_argument('--input', default='original', help='输入路径 (文件或目录)')
    parser.add_argument('--output', default='aligned_gt', help='输出路径 (文件或目录)')
    parser.add_argument('--single-file', action='store_true', help='处理单个文件而不是目录')
    
    args = parser.parse_args()
    
    if args.single_file:
        # 处理单个文件
        if not os.path.isfile(args.input):
            print(f"错误：输入文件不存在：{args.input}")
            return
        
        success = process_single_mesh(args.input, args.output)
        print(f"处理{'成功' if success else '失败'}")
    else:
        # 处理整个目录
        if not os.path.isdir(args.input):
            print(f"错误：输入目录不存在：{args.input}")
            return
            
        success_count, total_count = process_directory(args.input, args.output)
        print(f"\n处理完成！成功处理 {success_count}/{total_count} 个文件")

if __name__ == "__main__":
    main()
