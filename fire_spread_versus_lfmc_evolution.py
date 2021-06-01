# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:51:55 2021

@author: kkrao
"""

import os
import geopandas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from init import dir_data
import seaborn as sns

def sector_mask(matrix,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    shape = matrix.shape
    centre = (int(matrix.shape[0]/2), int(matrix.shape[1]/2))
    radius = centre[0]
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
    mask = circmask*anglemask
    matrix[~mask] = np.nan
    
    return matrix

def calc_fire_dist(matrix):
    x_axis = np.linspace(-1, 1, matrix.shape[0])[:, None]
    y_axis = np.linspace(-1, 1, matrix.shape[0])[None, :]
    arr = np.sqrt(x_axis ** 2 + y_axis ** 2)
    
    inner = np.array([0])
    outer = np.array([1])
    # arr.shape    
    # arr /= arr.max() ### dont need this because it makes the corner = 1 rather than radius = 1
    # arr = arr[:, :, None]
    arr = arr * outer + (1 - arr) * inner
    # plt.imshow(arr)
    
    return np.nanmax(matrix*arr)

def calc_grad(matrix):
    
    x_axis = np.linspace(-1, 1, matrix.shape[0])[:, None]
    y_axis = np.linspace(-1, 1, matrix.shape[0])[None, :]
    arr = np.sqrt(x_axis ** 2 + y_axis ** 2)
    # arr[arr>0.2] = np.nan
    # arr[~np.isnan(arr)] = 1
    
    # center = np.nanmean(matrix*arr) 
    
    arr = np.sqrt(x_axis ** 2 + y_axis ** 2)
    arr[arr<0.8] = np.nan
    arr[arr>1] = np.nan
    arr[~np.isnan(arr)] = 1
    
    edge = np.nanmean(matrix*arr)
    
    # plt.imshow(arr)
    return edge

    
gdf = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_events_clip.shp"))
gdf.ignition_d = pd.to_datetime(gdf.ignition_d)
gdf = gdf.loc[gdf.ignition_d>=pd.to_datetime("2016-05-01")]

angles = np.linspace(0,315,int(360/45))

columns = ["id","fireDay","pixels"]+["fireDist_%03d"%angle for angle in angles]+["lfmcDelta_%03d"%angle for angle in angles]

df = pd.DataFrame(columns = columns)
for filename in os.listdir(os.path.join(dir_data, "fire_events_evolution")):
    print("[INFO] Filename = %s"%filename)
    matrix = np.load(os.path.join(dir_data, "fire_events_evolution",filename))
    
    # if matrix.shape[1]<100:
    #     print("[INFO] scar size is %d pixels wide. Skipping."%matrix.shape[1])
    #     continue
    ###fire distance traveled
    # plt.imshow(matrix[0])
    for k in range(1, matrix.shape[2]-1): ## day zero or ignition day is just a point. Hence ignoring it. 
        fireRow = []
        for angle in angles:
            scar = sector_mask(np.copy(matrix[:,:,k+1]), (angle, angle+45))
            # plt.imshow(scar)
            maxDist = min(calc_fire_dist(scar),1.00)
            fireRow.append(maxDist)
        temp = np.array(fireRow).argsort()
        fireRanks = np.empty_like(temp)
        fireRanks[temp] = np.arange(len(temp))
        ##### LFMC grad
        
        lfmcRow = []
        # cumScar = np.sum(matrix[:,:,1:k], axis = 2)
        # cumScar[cumScar>0]=1
        for angle in angles:
            lfmc = sector_mask(matrix[:,:,0]*matrix[:,:,k+1], (angle, angle+45))
            # grad = calc_grad(lfmc)
            grad = np.nanmean(lfmc)
            lfmcRow.append(grad)
        
        temp = np.array(lfmcRow).argsort()
        lfmcRanks = np.empty_like(temp)
        lfmcRanks[temp] = np.arange(len(temp))
            
        toAppend = pd.DataFrame([[filename.split(".")[0], k, np.nansum(matrix[0])]+list(fireRanks*(np.array(fireRow)>0))+list(lfmcRanks*(np.array(lfmcRow)>0))], columns = df.columns)
        df = df.append(toAppend)
    # break

df.shape
df.head()
# df = df.replace(0,np.nan)
# df.to_excel(os.path.join(dir_data,"lfmc_fire_evolution.xlsx"))

# for index, row in df.iterrows():
#     x = row.loc[xCols]
#     y = row.loc[yCols]
#     ax.scatter(x,y,alpha = 0.01,color = "k",linewidth = 0)

# ax.set_xlabel("$\Delta$ LFMC rank")
# ax.set_ylabel("Fire dist. rank")
# plt.show()

# size filter

# df = df.loc[df.pixels>=3000].copy()

################

xCols = [col for col in df.columns if "lfmcDelta" in col]
yCols = [col for col in df.columns if "fireDist" in col]

x = df.loc[:,xCols].values.flatten()
y = df.loc[:,yCols].values.flatten()

order = pd.DataFrame({"lfmc":x,"fire":y})
order.replace(0,np.nan, inplace = True)
order.dropna(inplace = True)

fig, ax = plt.subplots()
sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "grey",fliersize = 0)

## subsetting 
df.id = df.id.astype("float32")
df = pd.merge(df, gdf.loc[:,["id","lc_name"]], on = "id")

# df.groupby("lc_name").id.count()

######### by individual lcs

lcFilter = ["Grasslands"]
sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
x = sdf.loc[:,xCols].values.flatten()
y = sdf.loc[:,yCols].values.flatten()
order = pd.DataFrame({"lfmc":x,"fire":y})
fig, ax = plt.subplots()
sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "limegreen",fliersize = 0)

lcFilter = ["Deciduous Broadleaf Forests","Evergreen Broadleaf Forests","Evergreen Needleleaf Forests","Mixed Forests"]
sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
x = sdf.loc[:,xCols].values.flatten()
y = sdf.loc[:,yCols].values.flatten()
order = pd.DataFrame({"lfmc":x,"fire":y})
fig, ax = plt.subplots()
sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "green",fliersize = 0)

lcFilter = ["Closed Shrublands","Open Shrublands","Savannas","Woody Savannas"]
sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
x = sdf.loc[:,xCols].values.flatten()
y = sdf.loc[:,yCols].values.flatten()
order = pd.DataFrame({"lfmc":x,"fire":y})
fig, ax = plt.subplots()
sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "brown",fliersize = 0)

fig, ax = plt.subplots()
df.pixels.hist(bins = 100,ax=ax)
ax.set_xlim(0,10000)

#%% 

# order.shape

# # fig,ax = plt.subplots()
# # sns.kdeplot(data=order, x="lfmc",y="fire",fill=True,ax=ax)

# order=order.loc[order.lfmc>5]
# fig, ax = plt.subplots()
# ax.scatter(order.lfmc,order.fire,alpha=0.1,s=1)
# ax.set_xlim(0,30)
