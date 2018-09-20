# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import model_renderer.syscall_renderer as renderer
from quat_math.transformations import random_quaternion
import os
import numpy as np
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import sys
import glob

def mute():
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY) 
    
def renderModel(model_filename, data_folder, num_model_imgs,
                mesh_dir_depth = 0,
                camera_dist = 2, 
                randomize_lighting = False):

    num_digits = len(str(num_model_imgs))
    [model_class, model] = model_filename.split('/')[(-3-mesh_dir_depth):(-1-mesh_dir_depth)]
    os.makedirs(os.path.join(data_folder, '{0}/{1}'.format(model_class, model)), exist_ok=True)

    filenames = []
    for j in range(num_model_imgs):
        filenames.append(os.path.join(data_folder, '{0}/{1}/{0}_{1}_{2:0{3}d}'.format(model_class, model, j, num_digits)))
     
    render_quats = []
    image_filenames = []
    
    for data_filename in filenames:
        image_filenames.append(data_filename + '.png')
        quat = random_quaternion()
        np.save(data_filename + '.npy',  quat)
        render_quats.append(quat)

    try:
        renderer.renderView(model_filename, render_quats,  camera_dist=camera_dist,
                   filenames=image_filenames, standard_lighting=not randomize_lighting)
    except:
        with open(os.path.join(data_folder,'{}_{}.txt'.format(model_class, model)), 'w') as f:
            f.write("%s\n" % sys.exc_info()[0])
    return filenames


def renderDataModelBatch(num_model_imgs, num_workers, 
                         model_filenames, data_folder, 
                         camera_dist = 2,
                         randomize_lighting = False,
                         mesh_dir_depth = 0):
    batch_render = partial(renderModel, 
                           data_folder = data_folder,
                           num_model_imgs = num_model_imgs,
                           mesh_dir_depth = mesh_dir_depth,
                           camera_dist = camera_dist, 
                           randomize_lighting = randomize_lighting)
                   
    pool = Pool(num_workers)#, initializer=mute)
    pbar = tqdm(pool.imap(batch_render, model_filenames), total=len(model_filenames))
    for j, filename in enumerate(pbar): 
        [model_class, model] = model_filenames[j].split('/')[-3:-1]
        model_image_path = os.path.join(data_folder, '{0}/{1}/*.png'.format(model_class, model))
        num_imgs = len(glob.glob(model_image_path))
        
        pbar.write('{}: Rendering {}:{}. Generated {} images'.format(j, model_class, model, num_imgs))
        pass
    
    return

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--models', type=str)
    parser.add_argument('--render_folder', type=str)
    
    parser.add_argument('--num_poses', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--randomize_lighting', dest='randomize_lighting', action='store_true')
    parser.add_argument('--camera_dist', type=float, default=2)
    parser.add_argument('--mesh_dir_depth', type=int, default=0)

    args = parser.parse_args()
    if(args.models[-4:] == '.txt'):
        with open(args.models, 'r') as f:
            args.model_filenames = f.read().split()
    elif(args.models[-4:] in ['.obj', '.ply']):
        args.model_filenames = [args.models]
    elif(os.path.isdir(args.models)):
        import glob
        args.model_filenames = []
        args.models = os.path.join(args.models, '')
        args.model_filenames.extend(glob.glob(args.models + '**/*.obj', recursive=True))
        args.model_filenames.extend(glob.glob(args.models + '**/*.ply', recursive=True))
    return args

def main(model_filenames,
         render_folder,
         num_poses = 500,
         num_workers = 32, 
         camera_dist = 2,
         randomize_lighting = False,
         mesh_dir_depth = 0):
    import IPython; IPython.embed()
    renderDataModelBatch(num_model_imgs = num_poses, 
                         num_workers = num_workers, 
                         model_filenames = model_filenames, 
                         data_folder = render_folder, 
                         camera_dist = camera_dist,
                         randomize_lighting = randomize_lighting,
                         mesh_dir_depth = mesh_dir_depth)

if __name__ == '__main__':
    args = getArgs()
    main(model_filenames = args.model_filenames,
         render_folder = args.render_folder,
         num_poses = args.num_poses,
         num_workers = args.num_workers, 
         camera_dist = args.camera_dist,
         randomize_lighting = args.randomize_lighting,
         mesh_dir_depth = args.mesh_dir_depth)
 
