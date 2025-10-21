# coding: utf-8
import os
import sys
import torch
import numpy as np
import json
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesAtlas,
    PointLights,
    BlendParams,
    TexturesUV,
    PerspectiveCameras,
)
import matplotlib.pyplot as plt
import matplotlib
import random
import argparse
from pathlib import Path
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

DEFAULT_PARAMS = {
    "image_size": 1024,
    "camera_dist": 3,
    "elev_angle": [0, 15, 30],
    "azim_angle": [0, 45, 315]
}

def normalize_vertices(vertices: torch.Tensor):
    min_coord = vertices.min(dim=0).values  
    max_coord = vertices.max(dim=0).values  

    center = (min_coord + max_coord) / 2.0
    vertices_centered = vertices - center  

    max_length = (max_coord - min_coord).max()
    vertices_normalized = vertices_centered / max_length  # scale to [-0.5, 0.5]

    return vertices_normalized

def get_mesh_texturesuv(obj_filename, device):
    verts, faces, aux = load_obj(
        obj_filename,
        device=device,
        load_textures=True
    )
    verts_uvs = aux.verts_uvs  # (Vt, 2)
    faces_uvs = faces.textures_idx  # (F, 3) 
    faces_uvs = faces_uvs.unsqueeze(0)  # (1, F, 3)
    texture_image = aux.texture_images

    if len(texture_image) == 0:
        raise ValueError(f"Object file {obj_filename} does not contain any textures.")

    texture_map = list(texture_image.values())[0].to(device)[None]  # (1, H, W, 3)
    textures = TexturesUV(
        maps=texture_map,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs[None]  
    )

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=textures,
    )
    return mesh

def get_renderer(image_size, device):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=3,
    )
    
    # 修改为3维RGB背景色
    #blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))  # RGB，设置背景为白色
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device)
    )
    return renderer

def render_image(renderer, mesh, mesh_name, azim, elev, output_dir):

    dist = DEFAULT_PARAMS["camera_dist"]
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    fov_degrees = 30
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = 1.0 / np.tan(fov_radians / 2)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=((focal_length, focal_length),),  # (fx, fy)
        principal_point=((0, 0),),  # (cx,cy)
        R=R,
        T=T
        )

    lights = PointLights(device=device, location=T)

    renderer.rasterizer.cameras = cameras
    renderer.shader.cameras = cameras
    renderer.shader.lights = lights

    image = renderer(mesh)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{mesh_name}_elev{int(elev)}_azim{int(azim)}.png"
    filepath = os.path.join(output_dir, filename)
    matplotlib.image.imsave(filepath, image[0, ..., :3].cpu().numpy())
    print(f"Rendering is completed: {filepath}")
    # 将RGB图像转换为RGBA
    # rgb_image = image[0, ..., :3].cpu().numpy()
    # # 创建alpha通道（根据RGB值判断是否为背景）
    # alpha = np.where(np.all(rgb_image == 1.0, axis=-1), 0, 1)
    # rgba_image = np.dstack((rgb_image, alpha))
    
    # plt.imsave(filepath, rgba_image, format='png')
    # print(f"Image has been saved to: {filepath}")
    return filepath

def compile_all_steps(image_size, device, elev, azim, mesh, output_dir, mesh_name):
    renderer = get_renderer(image_size, device)
    return render_image(renderer, mesh, mesh_name, azim, elev, output_dir)

def process_single_mesh(obj_path: str, output_dir: str, params: dict, random_select: bool = False) -> list:
    """处理单个mesh文件"""
    image_paths = []
    try:
        mesh = get_mesh_texturesuv(obj_path, device)
        mesh_name = Path(obj_path).stem
        
        for elev in params["elev_angle"]:
            for azim in params["azim_angle"]:
                filename = f"{mesh_name}_elev{int(elev)}_azim{int(azim)}.png"
                filepath = os.path.join(output_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"Skip rendering: {filepath}")
                else:
                    img_path = compile_all_steps(
                        params["image_size"],
                        device,
                        elev,
                        azim,
                        mesh,
                        output_dir,
                        mesh_name
                    )
                    image_paths.append(img_path)
                    
        return image_paths
    except Exception as e:
        print(f"Error processing {obj_path}: {str(e)}")
        return []

def process_directory(input_dir: str, output_dir: str, params: dict, random_select: bool = False, path_txt_file: str = None):
    """处理整个目录下的mesh文件"""
    processed_images = set()
    if random_select and path_txt_file and os.path.exists(path_txt_file):
        with open(path_txt_file, "r") as f:
            processed_images = {line.strip().strip('"').strip(',') for line in f.readlines()}
    
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        obj_filename = os.path.join(subdir_path, f"{subdir}_align.obj")
        
        if not os.path.exists(obj_filename):
            print(f".obj file not found: {obj_filename}")
            continue
            
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        image_paths = process_single_mesh(obj_filename, output_subdir, params)
        
        if random_select and image_paths and path_txt_file:
            selected_path = random.choice(image_paths)
            writein_path = os.path.abspath(selected_path)
            if writein_path not in processed_images:
                with open(path_txt_file, "a") as f:
                    f.write(f'"{writein_path}",\n')
                print(f"Random viewport saved to: {writein_path}")
            else:
                print(f"Skipping already saved viewport: {writein_path}")

def main():
    parser = argparse.ArgumentParser(description='Render 3D meshes from multiple viewpoints')
    parser.add_argument('--input', default='aligned_gt', help='Input directory containing .obj files')
    parser.add_argument('--output', default='rendered_gt', help='Output directory for rendered images')
    parser.add_argument('--random-select', action='store_true', help='Randomly select one viewport per object')
    parser.add_argument('--single-file', action='store_true', help='Process single file instead of directory')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, args.input)
    output_path = os.path.join(current_dir, args.output)
    path_txt_file = os.path.join(current_dir, 'rendered_gt.txt') if args.random_select else None
    
    os.makedirs(output_path, exist_ok=True)
    
    if args.single_file:
        if not os.path.isfile(input_path):
            print(f"Error: Input file not found: {input_path}")
            return
        process_single_mesh(input_path, output_path, DEFAULT_PARAMS)
    else:
        if not os.path.isdir(input_path):
            print(f"Error: Input directory not found: {input_path}")
            return
        process_directory(input_path, output_path, DEFAULT_PARAMS, args.random_select, path_txt_file)
    
    print("Rendering completed!")

if __name__ == "__main__":
    main()
