# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import os
from model_renderer.syscall_renderer import renderView
import quat_math.transformations as tf_trans

unit_test_folder = os.path.dirname(os.path.abspath(__file__))

def testSyscallRenderer():
    model_filename = os.path.join(unit_test_folder, 'obj_model/model_normalized.obj')
    image_filename = os.path.join(unit_test_folder, 'sys_obj.png') 
    quat = tf_trans.random_quaternion()
    
    print('\n========================')
    print('Testing Syscall Renderer')
    print('========================\n')
    print('Loading Model: {}'.format(model_filename))
    print('Saving Render: {}'.format(image_filename))
    renderView(model_filename, [quat], 
               filenames=[image_filename],
               standard_lighting=True)
    
    print('\n=============')
    print('== Success ==')
    print('=============\n')

if __name__ == '__main__':
    testSyscallRenderer()
  
