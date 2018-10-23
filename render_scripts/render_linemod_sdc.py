# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:14:47 2018

@author: bokorn
"""
import model_renderer.syscall_renderer as renderer
from quat_math.transformations import random_quaternion
from generic_pose.bbTrans.discretized4dSphere import S3Grid

import os
import numpy as np
from functools import partial
from multiprocessing import Pool
#from tqdm import tqdm
import sys
import glob

def mute():
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY) 
    
def renderModel(model_filename, model_scale,
                data_folder, render_quats,
                camera_dist = 2.0, standard_lighting = True):

    num_digits = len(str(num_model_imgs))
    #model = model_filename.split('/')[-2]
    model = model_filename[-6:-4]
    os.makedirs(os.path.join(data_folder, model), exist_ok=True)

    image_filenames = []
    for j, quat in enumerate(render_quats):
        data_prefix = os.path.join(data_folder, '{0}/linemod_{0}_{1:0{2}d}'.format(model, j, num_digits))
        np.save(data_prefix + '.npy',  quat)
        image_filenames.append(data_prefix + '.png')
    
    try:
        renderer.renderView(model_filename, render_quats,
                            camera_dist=camera_dist, model_scale = model_scale, 
                            filenames=image_filenames, standard_lighting = standard_lighting)
    except:
        with open(os.path.join(data_folder,'linemod_{}.txt'.format(model)), 'w') as f:
            f.write("%s\n" % sys.exc_info()[0])
    return filenames


def renderDataModelBatch(base_level, num_workers, model_filenames, data_folder, model_scales):
    grid = S3Grid(base_level)
    base_vertices = np.unique(grid.vertices, axis = 0)

    batch_render = partial(renderModel,
                           data_folder = data_folder,
                           render_quats = base_vertices,
                           camera_dist = 2,
                           standard_lighting = -1)
                   
    pool = Pool(num_workers)#, initializer=mute)
    for j, filename in enumerate(pool.starmap(batch_render, zip(model_filenames, model_scales))):
        model = model_filenames[j].split('/')[-2]
        model_image_path = os.path.join(data_folder, '{}/*.png'.format(model))
        num_imgs = len(glob.glob(model_image_path))
        print('{}: Rendering {}. Generated {} images'.format(j, model, num_imgs))
        pass
    
    return

if __name__ == '__main__':
    from pysixd.inout import load_yaml
    base_level = 2
    print('Rendering Linemod Classes')
    #model_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/linemod.txt'
    #with open(model_file, 'r') as f:    
    #    linemod_filenames = f.read().split()
    models_folder = '/media/bokorn/ExtraDrive2/benchmark/linemod6DC/models/'
    models_info = load_yaml(os.path.join(models_folder, 'models_info.yml'))
    model_scales = []
    linemod_filenames = []
    for k,v in models_info.items():
        linemod_filenames.append(os.path.join(models_folder, 'obj_{:02d}.ply'.format(k)))
        model_scales.append(1/v['diameter'])

    #linemod_folder = '/ssd0/bokorn/data/renders/linemod'
    linemod_folder = '/media/bokorn/ExtraDrive2/renders/linemod6DC'
    os.makedirs(linemod_folder, exist_ok=True)
    renderDataModelBatch(base_level = base_level, num_workers=20, model_filenames=linemod_filenames, data_folder=linemod_folder, model_scales = model_scales)
    

