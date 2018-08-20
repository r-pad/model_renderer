# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:41:23 2018

@author: bokorn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import sys
import math
import random
import numpy as np
import os, tempfile, glob, shutil

import cv2
import bpy

render_root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, render_root_folder)
import transformations as tf_trans
blank_blend_file_path = os.path.join(render_root_folder, 'blank.blend') 

#Should be a system variable
#blender_executable_path = '/home/bokorn/src/blender/blender-2.79-linux-glibc219-x86_64/blender'
#blender_executable_path = 'blender'



def objCentenedCameraPos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def renderView(model_file, pose_quats, camera_dist, filenames = None, 
               standard_lighting=False, debug_mode=False):
    temp_dirname = tempfile.mkdtemp()

    assert filenames is None or len(pose_quats) == len(filenames), 'filenames must be None or same size as pose_quats, Expected {}, Got {}'.format(len(pose_quats), len(filenames))

    if(filenames is None):
        image_filenames = []
        if(type(pose_quats) == np.ndarray):
            num_quats = pose_quats.shape[0]
        else:
            num_quats = len(pose_quats)

        num_digits = len(str(num_quats))
        for j in range(num_quats):
            image_filenames.append(os.path.join(temp_dirname, '{0:0{1}d}.png'.format(j,num_digits)))
    else:
        image_filenames = filenames
        
    path_file = temp_dirname + '/paths.txt'
    with open(path_file, 'w') as f:
        for filename in image_filenames:
            f.write('{}\n'.format(filename))

    if(standard_lighting):
        light_num_lower = 10 
        light_num_upper = 10
        light_energy_mean = 2
        light_energy_std = 1e-100
        light_environment_energy_lower = 10.0 
        light_environment_energy_upper = 10.0
    else:
        light_num_lower = 1
        light_num_upper = 6
        light_energy_mean = 2
        light_energy_std = 2
        light_environment_energy_lower = 0.5
        light_environment_energy_upper = 1

    if(len(pose_quats) > 2):
        quats_file = temp_dirname + '/quats.npy'
        np.save(quats_file, pose_quats)
        pose_quats = None
    else:
        quats_file = None
    try:
        main(shape_file = model_file,
             pose_quats = pose_quats,
             quats_file = quats_file,
             camera_dist = camera_dist,
             path_file = path_file, 
             light_num_lower = light_num_lower,
             light_num_upper = light_num_upper,
             light_energy_mean = light_energy_mean,
             light_energy_std = light_energy_std,
             light_environment_energy_lower = light_environment_energy_lower,
             light_environment_energy_upper = light_environment_energy_upper)

        image_filenames = sorted(glob.glob(temp_dirname+'/*.png'))
        
        rendered_imgs = []
        
        if(filenames is None):
            for render_filename in image_filenames:
                img = cv2.imread(render_filename, cv2.IMREAD_UNCHANGED)
                rendered_imgs.append(img)
                
        shutil.rmtree(temp_dirname)
        
    except Exception as e:
        shutil.rmtree(temp_dirname)
        raise(e)
        
    return rendered_imgs

def main(shape_file,
         pose_quats,
         quats_file,
         camera_dist,
         path_file,
         light_num_lower=1,
         light_num_upper=6,
         light_dist_lower=8,
         light_dist_upper=20,
         light_azimuth_lower=0,
         light_azimuth_upper=360,
         light_elevation_lower=-90,
         light_elevation_upper=90,
         light_energy_mean=2,
         light_energy_std=2,
         light_environment_energy_lower=0.5,
         light_environment_energy_upper=1):
    bpy.ops.wm.open_mainfile(filepath=blank_blend_file_path)
    bpy.context.scene.cycles.device = 'GPU'   
    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    # Input parameters
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False
    bpy.ops.object.delete()
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)

    bpy.ops.object.delete()
    bpy.ops.import_scene.obj(filepath=shape_file) 
    
    #bpy.context.scene.render.use_shadows = False
    #bpy.context.scene.render.use_raytrace = False
    
    #m.subsurface_scattering.use = True

    camObj = bpy.data.objects['Camera']
    # camObj.data.lens_unit = 'FOV'
    # camObj.data.angle = 0.2
        
    # set lights
    bpy.ops.object.select_all(action='TOGGLE')
    if 'Lamp' in list(bpy.data.objects.keys()):
        bpy.data.objects['Lamp'].data.energy = 0
        bpy.data.objects['Lamp'].select = True # remove default light
    bpy.ops.object.delete()

    if(pose_quats is not None):
        pose_quat_list = pose_quats
    elif(quats_file is not None):
        pose_quat_list = np.load(quats_file)
    else:
        raise AssertionError('pos_quats or quats_file is required')

    if path_file is None: 
        image_filenames = []
        num_quats = pose_quat_list.shape[0]
        num_digits = len(str(num_quats))
        for j in range(num_quats):
            image_filenames.append('{0:0{1}d}.png'.format(j,num_digits))
    else:
        with open(path_file, 'r') as f:
            image_filenames = f.read().split() 
            
    for pose_num, (pos_quat, filename) in enumerate(zip(pose_quat_list, image_filenames)):
        camera_mat = np.eye(4)
        camera_mat[0,3] = camera_dist
        #invert_z = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0], [0,0,0,1]])
        #quat_mat = tf_trans.quaternion_matrix(tf_trans.quaternion_inverse(pos_quat))
        pos_quat[2] *= -1.0
        quat_mat = tf_trans.quaternion_matrix(pos_quat)
        #view_mat = quat_mat.dot(z90.dot(camera_mat))
        #view_mat = z90.dot(quat_mat.dot(camera_mat))
        #view_mat = quat_mat.dot(invert_z.dot(camera_mat))
        view_mat = quat_mat.dot(camera_mat)
        cam_pos = view_mat[:3,3]              
        cam_quat = tf_trans.quaternion_from_matrix(view_mat)
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)
    
        # set environment lighting
        #bpy.context.space_data.context = 'WORLD'
        bpy.context.scene.world.light_settings.use_environment_light = True
        bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(light_environment_energy_lower, light_environment_energy_upper)
        bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
    
        # set point lights
        for i in range(random.randint(light_num_lower,light_num_upper)):
            light_azimuth_deg = np.random.uniform(light_azimuth_lower, light_azimuth_upper)
            light_elevation_deg  = np.random.uniform(light_elevation_lower, light_elevation_upper)
            light_dist = np.random.uniform(light_dist_lower, light_dist_upper)
            lx, ly, lz = objCentenedCameraPos(light_dist, light_azimuth_deg, light_elevation_deg)
            bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
            bpy.data.objects['Point'].data.energy = np.random.normal(light_energy_mean, light_energy_std)
            
        ro_mat_pre = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        ro_mat_post = np.array([[0,0,1],[0,-1,0],[1,0,0]])
        cam_rot = ro_mat_pre.dot(view_mat[:3,:3].T.dot(ro_mat_post))
        cam_mat = np.eye(4)       
        cam_mat[:3,:3] = cam_rot
        cam_quat = tf_trans.quaternion_from_matrix(cam_mat)
        
        camObj.location[0] = cam_pos[0]
        camObj.location[1] = cam_pos[1]
        camObj.location[2] = cam_pos[2]
        camObj.rotation_mode = 'QUATERNION'
        # Blender is [w,x,y,z]
        # Transform is [x,y,z,w]
        camObj.rotation_quaternion[0] = cam_quat[0]
        camObj.rotation_quaternion[1] = cam_quat[1]
        camObj.rotation_quaternion[2] = cam_quat[2]
        camObj.rotation_quaternion[3] = cam_quat[3]

        bpy.data.scenes['Scene'].render.filepath = filename
        bpy.ops.render.render( write_still=True )
