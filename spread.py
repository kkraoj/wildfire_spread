# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:37:47 2021

@author: kkrao
"""


import geopandas
import pandas as pd
import numpy as np
from init import dir_data
import matplotlib.pyplot as plt
import os


gdf = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_events_clip.shp"))
gdf.ignition_d = pd.to_datetime(gdf.ignition_d)

gdf = gdf.loc[gdf.ignition_d.dt.year>=2016]


gdf = gdf.loc[gdf.total_area>=10]

# timeStamps = pd.DataFrame(gdf.groupby("id").pixels.count())
# timeStamps.columns = ["timeStamps"]
# gdf = pd.merge(gdf,timeStamps, on = "id")

# gdf = gdf.loc[gdf.timeStamps>=2]


fig, ax = plt.subplots(figsize = (3,3))
gdf.total_area.hist(bins = 1000, ax= ax)
ax.set_xlim(0,200)


gdf.head()
gdf.shape
gdf.columns

    
    
def get_value(filename, mx, my, band = 1):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray().astype(np.float16)
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]
def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

matrix = np.random.rand(100,100)
mask = sector_mask(matrix.shape,(50,50),50,(315,50))
matrix[~mask] = 0
plt.imshow(matrix)
plt.show()


