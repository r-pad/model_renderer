# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""
import numpy as np
import shutil
import tempfile
import cv2
import sys
import os
import plyfile
import uuid
from tqdm import tqdm

from multiprocessing import Lock
import quat_math.transformations as tf_trans

# Depth Includes
import OpenEXR
import array
import Imath

render_root_folder = os.path.dirname(os.path.abspath(__file__))
blank_blend_file_path = os.environ.get('BLANK_BLEND_PATH', 
        os.path.join(render_root_folder, 'blank.blend'))
temp_render_dir = os.environ.get('TEMP_RENDER_DIR', None)

def identityFunc(x):
    return x

def mute():
    # redirect output to log file
    #logfile = 'blender_render.log'
    #open(logfile, 'a').close()
    open(os.devnull, 'w').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    #os.open(logfile, os.O_WRONLY)
    os.open(os.devnull, os.O_WRONLY)
    return old

def unmute(old):
    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

def objCentenedCameraPos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * np.pi
    theta = float(azimuth_deg) / 180 * np.pi
    x = (dist * np.cos(theta) * np.cos(phi))
    y = (dist * np.sin(theta) * np.cos(phi))
    z = (dist * np.sin(phi))
    return (x, y, z)

# from https://github.com/TonythePlaneswalker/pcn/blob/master/render/process_exr.py
def loadEXR(filename, height, width):
    file = OpenEXR.InputFile(filename)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth

class BpyRenderer(object):
    #stdout = mute()
    import bpy
    import mathutils
    #unmute(stdout)
    def __init__(self, blend_file_path = blank_blend_file_path,
                 resolution = (448,448), transform_func = identityFunc,
                 remove_doubles = True, edge_split = True):
        self.remove_doubles = remove_doubles
        self.edge_split = edge_split
        self.render_mutex = Lock()
        self.transformFunc = transform_func
        self.models = {}
        stdout = mute()
        self.temp_dir = tempfile.mkdtemp(dir = temp_render_dir)
        unmute(stdout)
        self.bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        self.bpy.data.scenes['Scene'].render.use_raytrace = False
        self.bpy.data.scenes['Scene'].render.use_simplify = True
        self.bpy.data.scenes['Scene'].render.use_shadows = False
        self.bpy.data.scenes['Scene'].render.resolution_x = resolution[0]
        self.bpy.data.scenes['Scene'].render.resolution_y = resolution[1]
        self.bpy.context.scene.cycles.device = 'GPU'
        self.bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
        self.bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        self.setStandardLighting()
        self.cam_obj = self.bpy.data.objects['Camera']
        self.depth_output = None

    def setLighting(self, light_num_lower=5, light_num_upper=10,
                    light_dist_lower=8, light_dist_upper=20,
                    light_azimuth_lower=0, light_azimuth_upper=360,
                    light_elevation_lower=-90, light_elevation_upper=90,
                    light_energy_mean=2, light_energy_std=2,
                    light_environment_energy_lower=0.5,
                    light_environment_energy_upper=10.0):
        self.randomize_lighting = True
        self.light_num_lower = light_num_lower
        self.light_num_upper = light_num_upper
        self.light_dist_lower = light_dist_lower
        self.light_dist_upper = light_dist_upper
        self.light_azimuth_lower = light_azimuth_lower
        self.light_azimuth_upper = light_azimuth_upper
        self.light_elevation_lower = light_elevation_lower
        self.light_elevation_upper = light_elevation_upper
        self.light_energy_mean = light_energy_mean
        self.light_energy_std = light_energy_std
        self.light_environment_energy_lower = light_environment_energy_lower
        self.light_environment_energy_upper = light_environment_energy_upper

    def removeLighting(self):
        # clear default lights
        self.bpy.ops.object.select_by_type(type='LAMP')
        self.bpy.ops.object.delete(use_global=False)

    def setStandardLighting(self):
        self.randomize_lighting = False
        self.removeLighting()
        # set environment lighting
        self.bpy.context.scene.world.light_settings.use_environment_light = True
        self.bpy.context.scene.world.light_settings.environment_energy = 10.0
        self.bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
        # set point lights
        light_dist = 15
        light_positions = np.array([[ 1, 0, 0],
                                    [ 0, 1, 0],
                                    [ 0, 0, 1],
                                    [-1, 0, 0],
                                    [ 0,-1, 0],
                                    [ 0, 0,-1],
                                    [ 1, 1, 1],
                                    [ 1, 1,-1],
                                    [ 1,-1, 1],
                                    [ 1,-1,-1],
                                    [-1, 1, 1],
                                    [-1, 1,-1],
                                    [-1,-1, 1],
                                    [-1,-1,-1],
                                    ])*light_dist

        for lx, ly, lz in light_positions:
            #light_azimuth_deg = np.random.uniform(self.light_azimuth_lower, self.light_azimuth_upper)
            #light_elevation_deg  = np.random.uniform(self.light_elevation_lower, self.light_elevation_upper)
            #lx, ly, lz = objCentenedCameraPos(light_dist, light_azimuth_deg, light_elevation_deg)
            self.bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
            self.bpy.context.selected_objects[0].data.energy = 2.0
            #self.bpy.data.objects['Point'].data.energy = 2.0
       
    def randomizeLighting(self):
        self.removeLighting()
        # set environment lighting
        self.bpy.context.scene.world.light_settings.use_environment_light = True
        self.bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(self.light_environment_energy_lower, self.light_environment_energy_upper)
        self.bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
        # set point lights
        for i in range(np.random.randint(self.light_num_lower,self.light_num_upper+1)):
            light_azimuth_deg = np.random.uniform(self.light_azimuth_lower, self.light_azimuth_upper)
            light_elevation_deg  = np.random.uniform(self.light_elevation_lower, self.light_elevation_upper)
            light_dist = np.random.uniform(self.light_dist_lower, self.light_dist_upper)
            lx, ly, lz = objCentenedCameraPos(light_dist, light_azimuth_deg, light_elevation_deg)
            self.bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
            self.bpy.context.selected_objects[0].data.energy = np.random.normal(self.light_energy_mean, self.light_energy_std)

    def loadModel(self, model_filename, model_scale=1.0, emit = 1.0):
        #self.bpy.ops.object.select_by_type(type='MESH')
        #self.bpy.ops.object.delete(use_global=False)

        model_file_ext = model_filename.split('.')[-1]
        if(model_file_ext.lower() == 'obj'):
            stdout = mute()
            self.bpy.ops.import_scene.obj(filepath=model_filename)
            unmute(stdout)
        elif(model_file_ext.lower() == 'ply'):
            stdout = mute()
            self.bpy.ops.import_mesh.ply(filepath=model_filename)
            #self.loadPly(model_filename)
            #self.bpy.context.selected_objects[-1].select = False
            unmute(stdout)
            mesh = self.bpy.context.selected_objects[0]
            if(model_scale != 1.0):
                mesh.scale[0] = model_scale
                mesh.scale[1] = model_scale
                mesh.scale[2] = model_scale
            mat = self.bpy.data.materials.new('material_1')
            mesh.active_material = mat
            mat.use_vertex_color_paint = True
            mat.diffuse_intensity = 1.0
            mat.specular_intensity = 0.0
            mat.emit = emit
        else:
            raise ValueError('Invalid Model File Type {}'.format(model_file_ext))

        # Pulled from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
        for bpy_object in self.bpy.context.selected_objects: #bpy.context.scene.objects:
            #if bpy_object.type in ['CAMERA', 'LAMP']:
            #    continue
            self.bpy.context.scene.objects.active = bpy_object
            if self.remove_doubles:
                self.bpy.ops.object.mode_set(mode='EDIT')
                self.bpy.ops.mesh.remove_doubles()
                self.bpy.ops.object.mode_set(mode='OBJECT')
            if self.edge_split:
                self.bpy.ops.object.modifier_add(type='EDGE_SPLIT')
                self.bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
                self.bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")


        model_id = uuid.uuid4()
        self.models[model_id] = self.bpy.context.selected_objects
        return model_id

    def loadPly(self, model_filename):
        self.bpy.ops.import_mesh.ply(filepath=model_filename)
        data = plyfile.PlyData.read(model_filename)
        verts = np.array([list(v) for v in data.elements[0].data])                                    
        rgba = np.hstack([verts[:,-3:]/255.0, np.ones((verts.shape[0],1))])
        obj = self.bpy.context.selected_objects[0]
        mesh = obj.data
        self.bpy.context.scene.objects.active = obj
        obj.select = True
        if mesh.vertex_colors:
            vcol_layer = mesh.vertex_colors.active
        else:
            vcol_layer = mesh.vertex_colors.new()

        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                loop_vert_index = mesh.loops[loop_index].vertex_index
                vcol_layer.data[loop_index].color = rgba[loop_vert_index]

        mat = self.bpy.data.materials.new(obj.name + '_mat')
        mat.use_vertex_color_paint = True 
        mat.use_vertex_color_light = True
        obj.data.materials.append(mat)

    def hideModel(self, model_id, hide=True):
        for obj in self.models[model_id]:
            try:
                obj.hide_render = hide 
            except ReferenceError:
                pass

    def hideAll(self, hide=True):
        for model in self.models.values():
            for obj in model:
                try:
                    obj.hide_render = hide
                except ReferenceError:
                    pass

    def deleteModel(self, model_id):
        self.bpy.ops.object.select_all(action='DESELECT')
        for obj in self.models[model_id]:
            try:
                obj.select = True
            except ReferenceError:
                pass
        self.bpy.ops.object.delete(use_global=False)
        del self.models[model_id]

    def deleteAll(self):
        self.bpy.ops.object.select_all(action='DESELECT')
        for model in self.models.values():
            for obj in model:
                try:
                    obj.select = True
                except ReferenceError:
                    pass
        self.bpy.ops.object.delete(use_global=False)
        self.models = {}

    def setCameraMatrix(self, fx, fy, px, py, w, h):
        self.bpy.context.scene.render.resolution_x = w
        self.bpy.context.scene.render.resolution_y = h
        self.bpy.context.scene.render.resolution_percentage = 100
        f = self.cam_obj.data.lens
        self.cam_obj.data.sensor_width =  w * f / fx
        self.cam_obj.data.sensor_height = h * f / fy
        self.cam_obj.data.shift_x = (w/2 - px)/w
        self.cam_obj.data.shift_y = (py - h/2)/h
        self.focal_length = (fx, fy)
        self.width = w
        self.height = h

    def setDepth(self): 
        # https://github.com/TonythePlaneswalker/pcn/blob/master/render/render_depth.py 
        # camera

        self.cam_obj.data.angle = np.arctan(self.width / 2. / self.focal_length[1]) * 2.

        # render layer
        scene = self.bpy.context.scene
        scene.render.filepath = 'buffer'
        #scene.render.image_settings.color_depth = '16'

        # compositor nodes
        scene.use_nodes = True
        tree = scene.node_tree
        rl = tree.nodes.new('CompositorNodeRLayers')
        self.depth_output = tree.nodes.new('CompositorNodeOutputFile')
        self.depth_output.base_path = ''
        self.depth_output.format.file_format = 'OPEN_EXR'
        tree.links.new(rl.outputs['Depth'], self.depth_output.inputs[0])

    def render(self, render_prefix = None, return_images = True):
        if(render_prefix is None):
            render_prefix = os.path.join(self.temp_dir, '#')

        try:
            render_filename = render_prefix + '.png'
            self.bpy.data.scenes['Scene'].render.filepath = render_filename
            if(self.depth_output is not None):
                depth_filename = render_prefix + '.exr'
                self.depth_output.file_slots[0].path = depth_filename
                depth_filename = depth_filename.replace('#', '1') 
                
            stdout = mute()
            self.bpy.ops.render.render( write_still=True )
            unmute(stdout)
            if(return_images):
                img = cv2.imread(render_filename, cv2.IMREAD_UNCHANGED)
                if(self.depth_output is not None):
                    depth = loadEXR(depth_filename, self.height, self.width)
                    return img, depth
                return img

        except Exception as e:
            raise(e)
        finally:
            if(return_images):
                os.remove(render_filename)
                if(self.depth_output is not None):
                    os.remove(depth_filename)

    def renderTrans(self, obj_mat, z_out = True):
        with self.render_mutex:
            cam_mat = np.linalg.inv(obj_mat)
            if(z_out):
                rotate_x = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                cam_mat = cam_mat.dot(rotate_x)
                
            self.cam_obj.matrix_world = self.mathutils.Matrix(cam_mat)
            return self.render()

    def renderPose(self, pose_quats, 
                   image_filenames=None, 
                   camera_dist = 2.0):
        with self.render_mutex:
            return_images = False
            try:
                if(image_filenames is None):
                    assert image_filenames is None or len(pose_quats) == len(image_filenames), 'image_filenames must be None or same size as pose_quats, Expected {}, Got {}'.format(len(pose_quats), len(filenames))
                    return_images = True
                    image_filenames = []
                    if(type(pose_quats) == np.ndarray):
                        num_quats = pose_quats.shape[0]
                    else:
                        num_quats = len(pose_quats)
            
                    num_digits = len(str(num_quats))
                    for j in range(num_quats):
                        image_filenames.append(os.path.join(self.temp_dir, '{0:0{1}d}.png'.format(j,num_digits)))

                camera_mat = np.eye(4)
                camera_mat[0,3] = camera_dist
                pbar = tqdm(enumerate(zip(pose_quats, image_filenames)), total = len(image_filenames))
                for pose_num, (pos_quat, filename) in pbar:
                    if(self.randomize_lighting):
                        self.randomizeLighting()

                    pos_quat = self.transformFunc(pos_quat.copy())
                    pos_quat[2] *= -1.0
                    quat_mat = tf_trans.quaternion_matrix(pos_quat)
                    view_mat = quat_mat.dot(camera_mat)
                    cam_pos = view_mat[:3,3]              
                    cam_quat = tf_trans.quaternion_from_matrix(view_mat)
                       
                    ro_mat_pre = np.array([[1,0,0],[0,0,1],[0,-1,0]])
                    ro_mat_post = np.array([[0,0,1],[0,-1,0],[1,0,0]])
                    cam_rot = ro_mat_pre.dot(view_mat[:3,:3].T.dot(ro_mat_post))
                    cam_mat = np.eye(4)       
                    cam_mat[:3,:3] = cam_rot
                    cam_quat = tf_trans.quaternion_from_matrix(cam_mat)
                    
                    self.cam_obj.location[0] = cam_pos[0]
                    self.cam_obj.location[1] = cam_pos[1]
                    self.cam_obj.location[2] = cam_pos[2]
                    self.cam_obj.rotation_mode = 'QUATERNION'
                    # Blender is [w,x,y,z]
                    # Transform is [x,y,z,w]
                    self.cam_obj.rotation_quaternion[0] = cam_quat[0]
                    self.cam_obj.rotation_quaternion[1] = cam_quat[1]
                    self.cam_obj.rotation_quaternion[2] = cam_quat[2]
                    self.cam_obj.rotation_quaternion[3] = cam_quat[3]

                    self.render(filename[:-4], return_images)

                if(return_images):
                    rendered_imgs = []
                    for render_filename in image_filenames:
                        img = cv2.imread(render_filename, cv2.IMREAD_UNCHANGED)
                        rendered_imgs.append(img)
                    return rendered_imgs
            except Exception as e:
                raise(e)
            finally:
                if(return_images):
                    for render_filename in image_filenames:
                        os.remove(render_filename)

    def __del__(self):
        shutil.rmtree(self.temp_dir)

