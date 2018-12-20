# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""
import os
import numpy as np

def scaleObjModel(in_filename, out_filename, scale = None, max_size = None):
    pts = []
    with open(in_filename, 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            if(line[:2] == 'v '):
                pts.append([float(x) for x in line.split()[1:]])
    pts = np.array(pts)
    extent = np.max(pts, axis=0) - np.min(pts, axis=0)
    center = extent/2.0 + np.min(pts, axis=0)

    if(scale is None):
        if(max_size is not None):
            scale = max_size/np.max(extent)
        else:
            scale = 1.0

    with open(in_filename, 'r') as in_file:
        with open(out_filename, 'w') as out_file:
            lines = in_file.readlines()
            for line in lines:
                if(line[:2] == 'v '):
                    p = np.array([float(x) for x in line.split()[1:]])
                    p = scale*(p - center)
                    out_file.write('v {} {} {}\n'.format(p[0], p[1], p[2]))
                else:
                    out_file.write(line)

 
