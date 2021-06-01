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
from shapely.geometry import Point, Polygon


gdfEvents = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_events_clip.shp"))
gdfEvents.ignition_d = pd.to_datetime(gdfEvents.ignition_d)
gdfEvents = gdfEvents.loc[gdfEvents.ignition_d>=pd.to_datetime("2016-05-01")]

gdf = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_daily_clip.shp"))
gdf.date = pd.to_datetime(gdf.date)
gdf = gdf.loc[gdf.date>=pd.to_datetime("2016-05-01")]

# gdf = gdf.loc[gdf.total_area>=1]

# fig, ax = plt.subplots(figsize = (3,3))
# gdf.total_area.hist(bins = 1000, ax= ax)
# ax.set_xlim(0,200)

gdf.head()
gdf.shape
gdf.columns

lfmcDir =  r"D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\lfmc"

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

def create_x_y_geopandas(mx,my):
    _pnts = []
    row=[]
    col=[]
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            _pnts.append(Point(mx[i,j], my[i,j]))
            row.append(i)
            col.append(j)
    pnts = geopandas.GeoDataFrame(geometry=_pnts)
    pnts['row']=row
    pnts['col']=col
    return pnts

def create_scar(pnts,shape):
    dimension = int(np.sqrt(pnts.shape[0]))
    scar = np.zeros((dimension,dimension))
    mask = pnts.within(shape)
    selectX= pnts.loc[(mask[mask==True]).index.values].row.values
    selectY= pnts.loc[(mask[mask==True]).index.values].col.values
    scar[selectX,selectY]=1
    return scar

ctr=0
for index, row in gdfEvents.iterrows():
    # if index<15723:
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
    
    if np.isnan(lfmc).mean()>0.5:
        print("[INFO] LFMC has %d%% nans"%(np.isnan(lfmc).mean()*100))
        continue
    # np.save(os.path.join(dir_data, "fire_events","lfmc","%d.npy"), lfmc)
    # plt.imshow(lfmc,cmap = "viridis")
    # plt.show()
    pnts = create_x_y_geopandas(mx,my)
    sgdf = gdf.loc[gdf.id==row.id].sort_values(by='date')
    data = np.zeros((mx.shape[0],mx.shape[1],sgdf.shape[0]+1)).astype(np.float32) #+1 because there is lfmc too
    data[:,:,0]=lfmc.copy() ##first band is lfmc
    sctr=1
    for sindex, srow in sgdf.iterrows():   
        scar = create_scar(pnts,srow.geometry)
        data[:,:,sctr]=scar.copy()
        sctr+=1
    ##first burn scar, then lfmc
    np.save(os.path.join(dir_data,"fire_events_evolution","%d.npy"%row.id ), data)
    # gg = np.load(os.path.join(dir_data,"fire_events","combined","%d.npy"%row.id ))
    ctr+=1
    # plt.imshow(scar,cmap = "magma")
    # plt.show()
    
    # break
print("[INFO] Total fires saved = %d"%ctr)