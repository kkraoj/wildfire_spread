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
from sklearn.ensemble import RandomForestRegressor

lc_dict = {'Croplands':'grass', 'Savannas':'shrubs', 'Grasslands':'grass', 'Woody Savannas':'shrubs',
       'Deciduous Broadleaf Forests':'forest', 'Open Shrublands':'shrubs',
       'Evergreen Needleleaf Forests':'forest', 'Closed Shrublands':'shrubs', 'Barren':'grass',
       'Water Bodies':'grass', 'Cropland/Natural  Vegetation  Mosaics':'grass',
       'Mixed Forests':'forest', 'Evergreen Broadleaf Forests':'forest',
       'Permanent Wetlands':'forest', 'Permanent Snow and Ice':'grass'}




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

def stacked(df):
    ndf = pd.DataFrame(columns = ["fireArea","lfmcCount","wind","elevation"])
    angles = np.linspace(0,315,int(360/45))
    for angle in angles:
        cols = [col for col in df.columns if "%03d"%angle in col]
        subset = df[cols].copy()
        subset = subset.rename(columns = dict(zip(cols,newCols)))
        ndf = ndf.append(subset, ignore_index = True)
    return ndf        
    
    

# lfmcThresh = {'grass':55,'shrubs':106,'forest':72}
lfmcThresh = 100
gdf = geopandas.read_file(os.path.join(dir_data,"CU_FIRED_events_clip.shp"))
gdf.ignition_d = pd.to_datetime(gdf.ignition_d)

gdf = gdf.loc[gdf.ignition_d>=pd.to_datetime("2016-05-01")]

angles = np.linspace(0,315,int(360/45))

columns = ["id","pixels","lfmcMean"]+["fireArea_%03d"%angle for angle in angles]\
    +["lfmcCount_%03d"%angle for angle in angles]\
        + ["wind_%03d"%angle for angle in angles]\
          +  ["elevation_%03d"%angle for angle in angles]

df = pd.DataFrame(columns = columns)
# for filename in ['76446.npy']:    
for filename in os.listdir(os.path.join(dir_data, "fire_events","fire_lfmc_wind_topo")):
    matrix = np.load(os.path.join(dir_data, "fire_events","fire_lfmc_wind_topo",filename))
    
    # if matrix.shape[1]<100:
        # prin1t("[INFO] scar size is %d pixels wide. Skipping."%matrix.shape[1])
        # continue
#     ###fire distance traveled
#     # plt.imshow(matrix[0])
    fireRow = []
    for angle in angles:
        scar = sector_mask(np.copy(matrix[0]), (angle, angle+45))
#         # plt.imshow(scar)
        fireArea = np.nansum(scar)
        fireRow.append(fireArea)
    temp = np.array(fireRow).argsort()
    fireRanks = np.empty_like(temp)
    fireRanks[temp] = np.arange(len(temp))
#     ##### LFMC grad
    lc = lc_dict[gdf.loc[gdf.id==int(filename.split('.')[0])].lc_name.values[0]]
    lfmcRow = []
    for angle in angles:
        lfmc = sector_mask(np.copy(matrix[1]), (angle, angle+45))
        lfmcLte = (lfmc<lfmcThresh).sum()
        lfmcRow.append(lfmcLte)
    
    temp = np.array(lfmcRow).argsort()
    lfmcRanks = np.empty_like(temp)
    lfmcRanks[temp] = np.arange(len(temp))


    real = np.nanmean(np.copy(matrix[3]))
    real = real-90
    newAngles = angles-real
    newAngles = np.remainder(newAngles,360)
    windRow = np.where(newAngles/180>=1,180 - np.remainder(newAngles,180), newAngles)
    windRanks = np.argsort(windRow)
            
    elevationRow = []
    for angle in angles:
        elevation = np.nanmean(sector_mask(np.copy(matrix[3]), (angle, angle+45)))
        # lfmcLte = (lfmc<lfmcThresh).sum()
        elevationRow.append(elevation)
    
    temp = np.array(elevationRow).argsort()
    elevationRanks = np.empty_like(temp)
    elevationRanks[temp] = np.arange(len(temp))
        
    # print(filename)
    toAppend = pd.DataFrame([[filename.split(".")[0], np.nansum(matrix[0]),np.nanmean(matrix[1])]+\
                             list(fireRanks*(np.array(fireRow)>0))+\
                                 lfmcRow+\
                                 list(windRanks*(np.array(windRow)>0))+\
                                 elevationRow],\
                            columns = df.columns)
    ## >0 because we dont want to artificially order two things which are equal (and = zero)
    df = df.append(toAppend)

#     # break

df.shape
df.head()

masterdf = df.copy()
# fig, ax = plt.subplots()
# df.pixels.hist(ax=ax,bins=1000)
# ax.set_xlim(0,500)

## size filter
df = masterdf.loc[masterdf.pixels>=500].copy() ## 1000 hectares
# df = df.replace(0,np.nan)
# df = df.replace(7,np.nan)
# df = masterdf.loc[masterdf.lfmcMean>=70].copy()

################


xCols = [col for col in df.columns if "lfmcCount_" in col]
yCols = [col for col in df.columns if "fireArea" in col]

x = df.loc[:,xCols].values.flatten()
y = df.loc[:,yCols].values.flatten()

order = pd.DataFrame({"lfmc":x,"fire":y}).astype("float")
order.loc[order.lfmc>=2000,'lfmc'] = np.nan

order["fireBinary"] = "other"
fireThresh = 6
order.loc[order.fire<=1,"fireBinary"] = "Less fire"
order.loc[order.fire>fireThresh,"fireBinary"] = "More fire"

fig, ax = plt.subplots()
sns.kdeplot(data=order, x="lfmc", hue="fireBinary",ax=ax,common_norm = False)
ax.set_xlabel("Count (LFMC < threshold)")
ax.set_ylabel("Density")
ax.set_xlim(-200,2000)


# xCols = [col for col in df.columns if "wind_" in col]
# yCols = [col for col in df.columns if "fireArea" in col]

# x = df.loc[:,xCols].values.flatten()
# y = df.loc[:,yCols].values.flatten()

# order = pd.DataFrame({"wind":x,"fire":y})


# fig, ax = plt.subplots()
# sns.boxplot(x="wind",y="fire", data=order, ax = ax,color = "grey")
# ax.set_xlabel("Rank of wind")
# ax.set_ylabel("Rank of area burned")

# xCols = [col for col in df.columns if "elevation_" in col]
# yCols = [col for col in df.columns if "fireArea" in col]

# x = df.loc[:,xCols].values.flatten()
# y = df.loc[:,yCols].values.flatten()

# order = pd.DataFrame({"elevation":x,"fire":y})


# fig, ax = plt.subplots()
# sns.boxplot(x="elevation",y="fire", data=order, ax = ax,color = "grey")
# ax.set_xlabel("Rank of elevation")
# ax.set_ylabel("Rank of area burned")




# ## subsetting 
# df.id = df.id.astype("float32")
# df = pd.merge(df, gdf.loc[:,["id","lc_name"]], on = "id")

# # df.groupby("lc_name").id.count()

# ######### by individual lcs

# lcFilter = ["Grasslands"]
# sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
# x = sdf.loc[:,xCols].values.flatten()
# y = sdf.loc[:,yCols].values.flatten()
# order = pd.DataFrame({"lfmc":x,"fire":y})
# fig, ax = plt.subplots()
# sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "limegreen")

# lcFilter = ["Deciduous Broadleaf Forests","Evergreen Broadleaf Forests","Evergreen Needleleaf Forests","Mixed Forests"]
# sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
# x = sdf.loc[:,xCols].values.flatten()
# y = sdf.loc[:,yCols].values.flatten()
# order = pd.DataFrame({"lfmc":x,"fire":y})
# fig, ax = plt.subplots()
# sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "green")

# lcFilter = ["Closed Shrublands","Open Shrublands","Savannas","Woody Savannas"]
# sdf = df.loc[df.lc_name.isin(lcFilter)].copy()
# x = sdf.loc[:,xCols].values.flatten()
# y = sdf.loc[:,yCols].values.flatten()
# order = pd.DataFrame({"lfmc":x,"fire":y})
# fig, ax = plt.subplots()
# sns.boxplot(x="lfmc",y="fire", data=order, ax = ax,color = "brown")

# fig, ax = plt.subplots()
# df.pixels.hist(bins = 100,ax=ax)
# ax.set_xlim(0,10000)

#%% RF 
# ndf = stacked(df)
# regr = RandomForestRegressor()

# X = ndf.drop("fireArea",axis = 1)
# y = ndf["fireArea"]
# regr.fit(X,y)
# regr.score(X,y)
