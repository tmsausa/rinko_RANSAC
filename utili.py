import numpy as np
import pylab

from PIL import Image
from random import choice
from scipy import hypot
from scipy.linalg import inv
import cv2
import matplotlib.pyplot as plt

import os
from scipy import *
from numpy import *
from scipy.ndimage import *
import pylab
 


def _appendimages(im1,im2):
    """ return a new image that appends the two images side-by-side."""

    #select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))), axis=0)
    else:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))), axis=0)

    return concatenate((im1,im2), axis=1) 
def appendimages(im1, im2):
    """ Return a new concatenated images side-by-side """
    if ndim(im1) == 2:
        return _appendimages(im1, im2)
    else:
        imr = _appendimages(im1[:, :, 0], im2[:, :, 0])
        img = _appendimages(im1[:, :, 1], im2[:, :, 1])
        imb = _appendimages(im1[:, :, 2], im2[:, :, 2])
        return dstack((imr, img, imb))
    
def plot_points_list(im1,im2, consensus_list ):
    """ show a figure with lines joining the accepted matches in im1 and im2
        input: im1,im2 (images as arrays), locs1,locs2 (location of features),
        matchscores (as output from 'match'). 
       [x1,y1],[x2,y2]    <- consensus_list[i]
    """

    im12 = appendimages(im1,im2)
    
    im3=cv2.cvtColor(im12.astype(np.uint8), cv2.COLOR_GRAY2BGR)  

    
    cols1 = im1.shape[1]     
    
    for i in range(len(consensus_list)):
        
        x1=  consensus_list[i][0][0]
        y1=  consensus_list[i][0][1]
        
        x2=  consensus_list[i][1][0]
        y2=  consensus_list[i][1][1]
            
        cv2.line(im3,(x1,y1),(x2+cols1,y2),(0,255,255),1,cv2.LINE_AA)

    return im3
     
        

def get_points(cv_matches ,  kp_box,  kp_scene ):
    '''
        Return the corresponding points in both the images
    '''
    
    
    plist = []    
    ids = []
    for i in range (len( cv_matches)  ) :
        
        amatch=cv_matches[i]
        queryIdx = amatch.queryIdx  
        trainIdx = amatch.trainIdx
        
        y1 = int(kp_box[queryIdx].pt[1])
        x1 = int(kp_box[queryIdx].pt[0])
        
        y2 = int(kp_scene[trainIdx].pt[1])
        x2 = int(kp_scene[trainIdx].pt[0])

        
        plist.append ( [[x1,y1],[x2,y2]]   ) 
        ids.append(i)
    return plist 
