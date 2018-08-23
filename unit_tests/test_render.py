# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:14:47 2018

@author: bokorn
"""
from model_renderer.syscall_renderer import renderView
from quat_math.transformations import random_quaternion

def testRenderView():
    camera_dist = 2
    pose = random_quaternion()
    filenames = ['plane.png']
    model_file = 'plane/model_normalized.obj'

    pose_quats = [pose]
    renderView(model_file, pose_quats,  camera_dist=camera_dist filenames=filenames, 
               standard_lighting=True, debug_mode=False)
    
if __name__ == '__main__':
    testRenderView()
    
