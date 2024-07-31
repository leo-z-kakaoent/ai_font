import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist


def vgetter(img, n_anchor=8, blackcut=0):
    where_black = np.where(np.array(img)<(255-blackcut))
    endpoint = np.array([[128, 64]])
    points = np.stack(where_black,1)
    dist = cdist(endpoint, points, metric='euclidean')
    anchor = points[np.argsort(dist.reshape(-1))][:n_anchor]
    leftpoints = points[np.argsort(dist.reshape(-1))][n_anchor:]
    noerror = True
    len_near = 1
    while (len(leftpoints) > 0) & noerror & (len_near>0):
        try:
            dist = cdist(anchor, leftpoints, metric='euclidean')
            wherenear = np.where(np.min(dist,0) < 1.5)[0]
            len_near = len(wherenear)
            anchor = np.concatenate([anchor, leftpoints[wherenear,:]])
            leftpoints = np.delete(leftpoints, wherenear,axis=0)
        except:
            noerror = False
    npimg = np.array(img)
    npimg[anchor[:,0],anchor[:,1]] = 255
    return Image.fromarray(npimg) if len(leftpoints) >0 else None

def hgetter(img, n_anchor=8, blackcut=0):
    where_black = np.where(np.array(img)<(255-blackcut))
    endpoint = np.array([[64, 128]])
    points = np.stack(where_black,1)
    dist = cdist(endpoint, points, metric='euclidean')
    anchor = points[np.argsort(dist.reshape(-1))][:n_anchor]
    leftpoints = points[np.argsort(dist.reshape(-1))][n_anchor:]
    noerror = True
    len_near = 1
    while (len(leftpoints) > 0) & noerror & (len_near>0):
        try:
            dist = cdist(anchor, leftpoints, metric='euclidean')
            wherenear = np.where(np.min(dist,0) < 1.5)[0]
            len_near = len(wherenear)
            anchor = np.concatenate([anchor, leftpoints[wherenear,:]])
            leftpoints = np.delete(leftpoints, wherenear,axis=0)
        except:
            noerror = False
    npimg = np.array(img)
    npimg[anchor[:,0],anchor[:,1]] = 255
    return Image.fromarray(npimg) if len(leftpoints) >0 else None