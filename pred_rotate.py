import os
import numpy as np
import trimesh
import logging
import glob
from typing import Tuple, List, Optional, Dict
import argparse

class MeshProcessor:
    """网格处理类，包含所有与网格预处理相关的方法"""
    
    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        对网格进行归一化处理，将模型缩放到边界框 [-0.5, 0.5] 内，并将中心移到原点
        
        Args:
            mesh: 原始网格
        Returns:
            归一化后的网格
        """
        # 提取原始顶点
        original_vertices = mesh.vertices.copy()
        
        # 找到每个坐标轴的最小值和最大值
        min_coord = original_vertices.min(axis=0)
        max_coord = original_vertices.max(axis=0)
        
        # 计算中心点并平移
        center = (min_coord + max_coord) / 2.0
        vertices_centered = original_vertices - center
        
        # 等比例缩放
        max_length = (max_coord - min_coord).max()
        vertices_normalized = vertices_centered / max_length
        
        # 更新网格顶点
        mesh.vertices = vertices_normalized
        return mesh

    @staticmethod
    def load_mesh(mesh_path: str) -> trimesh.Trimesh:
        """
        加载网格文件并处理场景对象
        
        Args:
            mesh_path: 网格文件路径
        Returns:
            加载的网格对象
        """
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            return list(mesh.geometry.values())[0]
        return mesh

class MeshAligner:
    """网格对齐类，包含所有与对齐相关的方法"""
    
    def __init__(self, reference_mesh: trimesh.Trimesh):
        """
        初始化对齐器
        
        Args:
            reference_mesh: 参考网格
        """
        self.reference_mesh = reference_mesh
        self.ref_centroid, self.ref_eigenvalues, self.ref_eigenvectors = self._perform_pca(reference_mesh)

    def _perform_pca(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行PCA分析"""
        vertices = mesh.vertices
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        covariance_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        return centroid, eigenvalues[idx], eigenvectors[:, idx]

    def _get_volume_distribution(self, mesh: trimesh.Trimesh, centroid: np.ndarray, 
                               eigenvector: np.ndarray, num_bins: int = 10) -> np.ndarray:
        """计算体积分布"""
        vertices, faces = mesh.vertices, mesh.faces
        projections = np.dot(vertices - centroid, eigenvector)
        bins = np.linspace(projections.min(), projections.max(), num_bins + 1)
        volumes = np.zeros(num_bins)
        
        for face in faces:
            face_vertices = vertices[face]
            face_center = face_vertices.mean(axis=0)
            face_proj = np.dot(face_center - centroid, eigenvector)
            face_area = np.linalg.norm(np.cross(face_vertices[1] - face_vertices[0], 
                                              face_vertices[2] - face_vertices[0])) / 2
            bin_idx = np.digitize(face_proj, bins) - 1
            if 0 <= bin_idx < num_bins:
                volumes[bin_idx] += face_area
                
        return volumes

    def _get_direction_score(self, volumes: np.ndarray) -> float:
        """计算方向得分"""
        half = len(volumes) // 2
        front_volume = np.sum(volumes[:half])
        back_volume = np.sum(volumes[half:])
        return front_volume / (front_volume + back_volume)

    def align_pca(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        使用PCA进行粗对齐
        
        Args:
            mesh: 待对齐的网格
        Returns:
            对齐后的网格
        """
        mesh_centroid, _, mesh_eigenvectors = self._perform_pca(mesh)
        corrected_eigenvectors = mesh_eigenvectors.copy()
        
        for i in range(3):
            ref_volumes = self._get_volume_distribution(self.reference_mesh, self.ref_centroid, 
                                                      self.ref_eigenvectors[:, i])
            mesh_volumes = self._get_volume_distribution(mesh, mesh_centroid, mesh_eigenvectors[:, i])
            
            ref_score = self._get_direction_score(ref_volumes)
            mesh_score = self._get_direction_score(mesh_volumes)
            
            print(f'{i} 轴体积分布 - 参考模型得分: {ref_score:.3f}, 当前模型得分: {mesh_score:.3f}')
            
            if (mesh_score > 0.5) != (ref_score > 0.5):
                corrected_eigenvectors[:, i] *= -1
                print(f'轴 {i} 需要翻转')
        
        rotation_matrix = np.dot(self.ref_eigenvectors, corrected_eigenvectors.T)
        aligned_mesh = mesh.copy()
        aligned_mesh.apply_transform(np.vstack((
            np.hstack((rotation_matrix, np.zeros((3, 1)))),
            [0, 0, 0, 1]
        )))
        
        return aligned_mesh

    def align_icp(self, mesh: trimesh.Trimesh, n_points: int = 200000, 
                 max_iterations: int = 100) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        使用ICP进行精细对齐
        
        Args:
            mesh: 待对齐的网格
            n_points: 采样点数
            max_iterations: 最大迭代次数
        Returns:
            对齐后的网格和变换矩阵
        """
        try:
            source_points, _ = trimesh.sample.sample_surface(mesh, n_points)
            target_points, _ = trimesh.sample.sample_surface(self.reference_mesh, n_points)
            
            print(f"采样点数 - 源模型: {len(source_points)}, 目标模型: {len(target_points)}")
            
            M, transformed, cost = trimesh.registration.icp(source_points,
                                                          target_points,
                                                          initial=None,
                                                          max_iterations=max_iterations,
                                                          reflection=False)
            print(f"ICP对齐成本: {cost}")
            
            aligned_mesh = mesh.copy()
            aligned_mesh.apply_transform(M)
            return aligned_mesh, M
            
        except Exception as e:
            print(f"ICP对齐失败: {str(e)}")
            return mesh, np.eye(4)

def process_alignment_task(align_gt_path: str, compare_paths: List[str], output_dir: Optional[str] = None) -> None:
    """
    处理单个参考模型的对齐任务
    
    Args:
        align_gt_path: 参考模型路径
        compare_paths: 待对齐模型路径列表
        output_dir: 输出目录，如果不指定则使用参考模型所在目录
    """
    processor = MeshProcessor()
    
    # 加载并处理参考模型
    print(f"\n处理参考模型: {align_gt_path}")
    align_gt_mesh = processor.load_mesh(align_gt_path)
    
    # 初始化对齐器
    aligner = MeshAligner(align_gt_mesh)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(align_gt_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个比较模型
    for comp_path in compare_paths:
        print(f"\n处理模型 {comp_path}:")
        
        try:
            # 加载和预处理
            compare_gt = processor.load_mesh(comp_path)
            compare_gt = processor.normalize_mesh(compare_gt)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(comp_path))[0]
            ref_name = os.path.splitext(os.path.basename(align_gt_path))[0]
            
            # PCA对齐
            print("正在进行PCA对齐...")
            pca_aligned = aligner.align_pca(compare_gt)
            pca_path = os.path.join(output_dir, f"{base_name}_pca_aligned.glb")
            pca_aligned.export(pca_path)
            print(f"已保存PCA对齐结果到: {pca_path}")
            
            # ICP精细对齐
            print("正在进行ICP精细对齐...")
            icp_aligned, _ = aligner.align_icp(pca_aligned)
            icp_path = os.path.join(output_dir, f"{base_name}_icp_aligned.glb")
            icp_aligned.export(icp_path)
            print(f"已保存ICP对齐结果到: {icp_path}")
            
        except Exception as e:
            print(f"处理模型 {comp_path} 时发生错误: {str(e)}")
            continue

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='对齐和旋转3D网格模型')
    parser.add_argument('--gt-root', default='aligned_gt',
                      help='包含参考模型的根目录 (默认: aligned_gt)')
    parser.add_argument('--pred-root', default='trellis_pred',
                      help='包含预测模型的根目录 (默认: trellis_pred)')
    parser.add_argument('--output-root', default='trellis_rotated',
                      help='输出目录 (默认: trellis_rotated)')
    parser.add_argument('--single-model', default=None,
                      help='可选：只处理特定模型，例如"chair"')
    
    args = parser.parse_args()
    
    # 确保所有必要的目录都存在
    if not os.path.exists(args.gt_root):
        print(f"错误：参考模型目录不存在：{args.gt_root}")
        return
    if not os.path.exists(args.pred_root):
        print(f"错误：预测模型目录不存在：{args.pred_root}")
        return
    
    # 创建输出根目录
    os.makedirs(args.output_root, exist_ok=True)
    
    # 获取要处理的子目录列表
    if args.single_model:
        subdirs = [args.single_model]
        if not os.path.exists(os.path.join(args.gt_root, args.single_model)):
            print(f"错误：指定的模型 {args.single_model} 在 {args.gt_root} 中不存在")
            return
    else:
        subdirs = os.listdir(args.gt_root)
    
    print(f"\n开始处理模型对齐任务：")
    print(f"参考模型目录：{args.gt_root}")
    print(f"预测模型目录：{args.pred_root}")
    print(f"输出目录：{args.output_root}")
    print(f"找到 {len(subdirs)} 个待处理模型\n")
    
    for subdir in sorted(subdirs):
        print(f"\n{'='*80}")
        print(f"处理模型：{subdir}")
        print(f"{'='*80}")
        
        # 构建路径
        align_gt_path = os.path.join(args.gt_root, subdir, f"{subdir}_align.obj")
        compare_paths = glob.glob(os.path.join(args.pred_root, subdir, "*.glb"))
        output_dir = os.path.join(args.output_root, subdir)
        
        if not os.path.exists(align_gt_path):
            print(f"警告：参考模型文件不存在：{align_gt_path}")
            continue
            
        if not compare_paths:
            print(f"警告：在 {os.path.join(args.pred_root, subdir)} 中未找到.glb文件")
            continue
            
        print(f"找到 {len(compare_paths)} 个预测模型需要对齐")
        process_alignment_task(align_gt_path, compare_paths, output_dir)
    
    print("\n所有对齐任务完成！")

if __name__ == "__main__":
    main()