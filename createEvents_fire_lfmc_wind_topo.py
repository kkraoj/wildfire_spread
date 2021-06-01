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
import gdal
import shapely
import datetime

gdf = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_events_clip.shp"))
gdf.ignition_d = pd.to_datetime(gdf.ignition_d)

gdf = gdf.loc[gdf.ignition_d>=pd.to_datetime("2016-05-01")]


# gdf = gdf.loc[gdf.total_area>=1]

# fig, ax = plt.subplots(figsize = (3,3))
# gdf.total_area.hist(bins = 1000, ax= ax)
# ax.set_xlim(0,200)

# gdf.head()
# gdf.shape
# gdf.columns

lfmcDir =  r"D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\lfmc"
elevationFile = r"D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\Elevation\Elevation\usa_dem.tif"

def get_value(filename, mx, my, band = 1):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray().astype(np.float32)
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]

def get_radius_helper(line, center):
    xs = np.array(line.coords.xy[0])
    ys = np.array(line.coords.xy[1])
    
    radius = max(np.sqrt((xs-center[0])**2+(ys-center[1])**2))
    return radius

def get_radius(shapes, center):
    # shapes=row.geometry
    lines=shapes.boundary
    if type(shapes)==shapely.geometry.polygon.Polygon:
        if type(lines)==shapely.geometry.linestring.LineString:
            radius = get_radius_helper(lines, center)
        else:
            radius=0
            for line in lines:
                newRadius = get_radius_helper(line, center)
                if newRadius>radius:
                    radius = newRadius
    else:
        radius = 0
        for shape in shapes:
            lines=shape.boundary
            if type(lines)==shapely.geometry.linestring.LineString:
                newRadius = get_radius_helper(lines, center)
                if newRadius>radius:
                    radius = newRadius
            else:
                for line in lines:
                    newRadius = get_radius_helper(line, center)
                    if newRadius>radius:
                        radius = newRadius
    return radius

def create_x_y(center,radius, delta =0.002243):
    left = center[0]-radius
    right = center[0]+radius
    top = center[1]+radius
    bottom = center[1]-radius
    
    mx = np.linspace(left, right, num = int((right-left)/delta))
    my = np.linspace(bottom, top, num = int((top-bottom)/delta))
    
    mx,my = np.meshgrid(mx,my)
    
    return mx, my
ctr=0
for index, row in gdf.iterrows():
    # if index<17580:
        # continue
    print("[INFO] Index = %d"%index)
    # row = gdf.loc[15720]
    date = row.ignition_d
    
    if date.day>=15:
        day = 15
    else:
        day=1
    lfmcDate = "%04d-%02d-%02d"%(date.year, date.month, \
                               day)
    lfmcFile = os.path.join(lfmcDir, "lfmc_map_%s.tif"%lfmcDate)
    
    center = (row.ignition_2, row.ignition_l)
    if center == (0,0):
        print("[INFO] Center is (0,0)")
        continue
    radius = get_radius(row.geometry, center)
    mx,my = create_x_y(center,radius)   
    
    try:
        lfmc = get_value(lfmcFile, mx, my)
    except:
        print("[INFO] LFMC patch requested is out of bounds")
        continue
    
    lfmc[lfmc<0]=np.nan
    
    # if np.isnan(lfmc).mean()>0.5:
    #     print("[INFO] LFMC has %d%% nans"%(np.isnan(lfmc).mean()*100))
    #     continue
    # np.save(os.path.join(dir_data, "fire_events","lfmc","%d.npy"), lfmc)
    # plt.imshow(lfmc,cmap = "viridis")
    # plt.show()

    scarFile = os.path.join(dir_data, "fire_events","scar","%4d.tif"%date.year)
    try:
        scar = get_value(scarFile, mx, my)
    except:
        print("[INFO] scar patch requested is out of bounds")
        continue
    
    scar[scar>1e8] = np.nan
    scar[scar>0] = 1
    
    windFile = os.path.join(dir_data,"wind_direction","th_%4d.nc"%date.year)
    dayofyear = date.dayofyear
    try:
        wind = get_value(windFile, mx, my,dayofyear)
    except:    
        print("[INFO] wind patch requested is out of bounds")
        continue
    wind[scar>360] = np.nan
    
    try:
        elevation = get_value(elevationFile, mx, my)
    except:    
        print("[INFO] elevation patch requested is out of bounds")
        continue
    
    ##first burn scar, then lfmc
    data = np.stack([scar,lfmc,wind,elevation])
    np.save(os.path.join(dir_data,"fire_events","fire_lfmc_wind_topo","%d.npy"%row.id ), data)
    # gg = np.load(os.path.join(dir_data,"fire_events","combined","%d.npy"%row.id ))
    ctr+=1
    # plt.imshow(scar,cmap = "magma")
    # plt.show()
    
    # break
print("[INFO] Total fires saved = %d"%ctr)