# Use trimesh to convert .obj file to .glb file;
# Supports .obj format that contain only vertices (v) and faces (f) with embedded color and coordinate data.
import os
import trimesh
import argparse
# Generate texture for .obj to render in Blender, otherwise color info cannot be read by Blender
def fix_vert_color_glb(mesh_path):
    from pygltflib import GLTF2, Material, PbrMetallicRoughness
    obj1 = GLTF2().load(mesh_path)
    obj1.meshes[0].primitives[0].material = 0
    obj1.materials.append(Material(
        pbrMetallicRoughness = PbrMetallicRoughness(
            baseColorFactor = [1.0, 1.0, 1.0, 1.0],
            metallicFactor = 0.,
            roughnessFactor = 1.0,
        ),
        emissiveFactor = [0.0, 0.0, 0.0],
        doubleSided = True,
    ))
    obj1.save(mesh_path)
    
def convert_obj_to_glb(obj_path, output_path):
    """转换单个obj文件到glb格式"""
    mesh = trimesh.load(obj_path, process=False)
    mesh.export(output_path)
    fix_vert_color_glb(output_path)
    return output_path

def batch_convert_objs(input_root, output_root):
    """批量转换目录下的所有obj文件"""
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 遍历输入目录
    for subdir in os.listdir(input_root):
        input_subdir = os.path.join(input_root, subdir)
        if not os.path.isdir(input_subdir):
            continue
            
        # 创建对应的输出子目录
        #output_subdir = os.path.join(output_root, subdir)
        #os.makedirs(output_subdir, exist_ok=True)
        
        # 处理子目录中的所有obj文件
        for filename in os.listdir(input_subdir):
            if filename.endswith('.obj'):
                input_path = os.path.join(input_subdir, filename)
                output_filename = filename.replace('.obj', '.glb')
                output_path = os.path.join(output_root, output_filename)
                
                try:
                    print(f"Converting {input_path} to {output_path}")
                    convert_obj_to_glb(input_path, output_path)
                    print(f"Successfully converted {filename}")
                except Exception as e:
                    print(f"Error converting {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Render 3D meshes from multiple viewpoints')
    parser.add_argument('--input', default='aligned_gt', help='Input directory containing .obj files')
    parser.add_argument('--output', default='aligned_gt_glb' , help='Output directory for rendered images')

    
    args = parser.parse_args()
    input_root = args.input  # 输入目录
    output_root = args.output # 输出目录
    
    print(f"Starting batch conversion from {input_root} to {output_root}")
    batch_convert_objs(input_root, output_root)
    print("Conversion completed!")


if __name__ == "__main__":
    main()


