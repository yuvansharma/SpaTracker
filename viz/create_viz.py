import sys
import bpy
import trimesh
import argparse
import numpy as np 
from pathlib import Path 
import os 

from mathutils import Vector
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--track_len', type=int, default=25)
    parser.add_argument('--outdir', type=str, default='blender_viz')

    CAM_Z_OFFSET = 4
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    args = parser.parse_args(argv)

    outdir = args.outdir

    track_len = int(args.track_len)

    basename = args.input.split('/')[-1].split('.')[0]
    episode_name = basename
    data = np.load(args.input)
    xyz = data[...,:3]
    msk = (xyz[0,...,2] != 0)
    mean = xyz[0][msk].mean(axis=0)

    xyz[...,-2] *= -1
    xyz[...,-1] *= -1
    
    rgb = data[:,:,[3,4,5]].astype(np.uint8)
    rgb = np.concatenate([rgb, np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8) * 255],
                         axis=-1)

    pcd = trimesh.PointCloud(vertices=xyz[0], colors=rgb[0])
    os.makedirs(outdir, exist_ok=True)
    pcd.export(os.path.join(outdir, f"{episode_name}.ply"))

    ply_path = Path(os.path.join(outdir, f"{episode_name}.ply")).absolute()
    print(f"PLY file path: {ply_path}")

    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()


    bpy.data.objects['Camera'].location = (mean[0],mean[1],mean[2] + CAM_Z_OFFSET)
    bpy.data.objects['Camera'].rotation_euler = (0,0,0)

    # add a plane
    bpy.ops.mesh.primitive_plane_add(size=50, enter_editmode=False, align='WORLD', location=(0,0,xyz[...,2].min()), scale=(10, 10, 10))

    bpy.ops.wm.ply_import(filepath=str(ply_path))
    

    obj = bpy.data.objects[basename]
    bpy.context.view_layer.objects.active = obj

    geo_node_modifier = obj.modifiers.new(name='GeometryNodes', type='NODES')
    bpy.ops.node.new_geometry_node_group_assign()


    node_tree = geo_node_modifier.node_group
    group_input = node_tree.nodes['Group Input']
    group_output = node_tree.nodes['Group Output']

    mesh2points = node_tree.nodes.new(type='GeometryNodeMeshToPoints')
    
    mesh2points.inputs['Radius'].default_value=0.07
    if basename == 'colored_raw_points_example':
        mesh2points.inputs['Radius'].default_value=0.06 
    node_tree.links.new(group_input.outputs[0], mesh2points.inputs[0])

    setMaterial = node_tree.nodes.new(type='GeometryNodeSetMaterial')
    node_tree.links.new(mesh2points.outputs[0], setMaterial.inputs['Geometry'])
    node_tree.links.new(setMaterial.outputs['Geometry'], group_output.inputs[0])


    point_material = bpy.data.materials.new(name="PointMaterial")
    point_material.use_nodes = True

    if obj.data.materials:
        obj.data.materials[0] = point_material
    else:
        obj.data.materials.append(point_material)

    setMaterial.inputs['Material'].default_value = point_material

    MatNodeTree = point_material.node_tree

    MatNodeTree.nodes.clear()

    attribute_node = MatNodeTree.nodes.new(type='ShaderNodeAttribute')
    attribute_node.attribute_name = 'Col'
    attribute_node.attribute_type = 'GEOMETRY'

    shader_node = MatNodeTree.nodes.new(type='ShaderNodeBsdfPrincipled')
    MatNodeTree.links.new(attribute_node.outputs['Color'], shader_node.inputs['Base Color'])

    shader_node.inputs['Metallic'].default_value = 0.77

    shader_output = MatNodeTree.nodes.new(type='ShaderNodeOutputMaterial')
    MatNodeTree.links.new(shader_node.outputs['BSDF'], shader_output.inputs['Surface'])

    # set cycles as render engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    
    # set start and end frame
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = xyz.shape[0]
    
    if basename == 'colored_raw_points_example':
        bpy.data.objects['Camera'].location = (2, -1.27, 9)
        print("set custom camera location for colored raw points example")


    cmap = cm.get_cmap("gist_rainbow")
    tcolors = np.zeros((data.shape[0],data.shape[1],4))

    T, N = xyz.shape[:2]
    for t in range(T):	
        y_min = np.min(data[t,:,1])
        y_max = np.max(data[t,:,1])
        norm = plt.Normalize(y_min,y_max)
        for n in range(N):
            color = cmap(norm(data[t,n,1]))
            tcolors[t,n] = color

    tmap = tcolors
    tmap[...,-1]*=1.0

    def create_trajectory(i, length=5):
        name = 'Curve.{:04d}'.format(i)
        curve_data = bpy.data.curves.new(name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.fill_mode = 'FULL'
        curve_data.bevel_depth = 0.003
        curve_data.bevel_resolution = 3
        
        curve_object = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(curve_object)
        
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(length-1)
        mat = create_emission_material('Mat.{:04d}'.format(i),tmap[0,i],1)
        
        for _ in range(length):
            x, y, z = xyz[_,i]
            polyline.points[_].co = (0, 0, 0, 1)
        
        curve_object.data.materials.append(mat)
            
        return curve_object

    def create_emission_material(name, color, strength):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()

        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        
        emission.inputs['Strength'].default_value = strength

        material_output = nodes.new(type='ShaderNodeOutputMaterial')
        material_output.location = 400, 0

        links = material.node_tree.links
        link = links.new(emission.outputs['Emission'], material_output.inputs['Surface'])

        return material    

    for t in range(T):
        for i in range(N):
            x,y,z = xyz[t,i]
            obj.data.vertices[i].co = Vector((x,y,z))
            obj.data.vertices[i].keyframe_insert(data_path='co',frame=t)
            
    curve_list = [ create_trajectory(i, track_len) for i in range(N)]
    
    for t in range(T):
        for i, curve in enumerate(curve_list):
            for j in range(track_len):
                u = max(t-j,0)
                x,y,z = xyz[u,i,:3]
                if z == 0:
                    continue
                if j > 0:
                    if abs(curve.data.splines[0].points[j-1].co[2]-z)>0.5:
                        for k in range(j):
                            curve.data.splines[0].points[k].co = (x,y,z,1)
                            curve.data.splines[0].points[k].keyframe_insert(data_path='co',frame=t)
                curve.data.splines[0].points[j].co = (x,y,z,1)
                curve.data.splines[0].points[j].keyframe_insert(data_path='co',frame=t)        
            # filter out the points that are not visible in the first frame
    
    
    if os.path.exists(f'{episode_name}.blend'):
        os.remove(f'{episode_name}.blend')

    video_name = episode_name
    output_path = os.path.join(outdir, f"{video_name}_")

    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    bpy.context.scene.render.fps = 20
    bpy.context.scene.cycles.samples = 20
    
    num_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
    print(f"Starting render, total frames: {num_frames}")

    
    bpy.ops.render.render(animation=True)
    
    print(f"Output video saved. Exiting")

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, f'{episode_name}.blend'))
    # then quit
    bpy.ops.wm.quit_blender()
