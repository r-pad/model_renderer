# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import os
import model_renderer.pose_renderer as pose_renderer                                                 
import quat_math.transformations as tf_trans

unit_test_folder = os.path.dirname(os.path.abspath(__file__))

def testBpyRenderer():
    model_filename = os.path.join(unit_test_folder, 'obj_model/model_normalized.obj')
    image_filename = os.path.join(unit_test_folder, 'bpy_obj.png') 
    quat = tf_trans.random_quaternion()
    
    print('\n===================')
    print('Testing BpyRenderer')
    print('===================\n')
    rdr = pose_renderer.BpyRenderer()
    print('Loading Model: {}'.format(model_filename))
    rdr.loadModel(model_filename)
    img = rdr.renderPose([quat])[0]
    print('Returning Render: Size: {}, Min: {}, Max: {}'.format(img.shape, img.min(), img.max()))
    print('Saving Render: {}'.format(image_filename))
    rdr.renderPose([quat], [image_filename])
    
    print('\n=============')
    print('== Success ==')
    print('=============\n')

if __name__ == '__main__':
    testBpyRenderer()
 
