# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:40:45 2021

@author: kkrao
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from init import dir_data
import os


def update_time():
    t = -1
    t_max = matrix.shape[2]-2
    while 1:
        t += anim.direction
        yield t%(t_max+1)
    # while (t<t_max)&(t>=-1):
        # t += anim.direction
        # yield t

def update_plot(t):
    time_text.set_text(time_template%(t))
    image.set_data(matrix[:,:,t+1])
    return time_text

def on_press(event):
    if event.key.isspace():
        if anim.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()
        anim.running ^= True
    elif event.key == 'left':
        anim.direction = -1
    elif event.key == 'right':
        anim.direction = +1

    # Manually update the plot
    if event.key in ['left','right']:
        t = anim.frame_seq.__next__()
        update_plot(t)
        plt.draw()

filenames = []
for filename in os.listdir(os.path.join(dir_data, "fire_events_evolution")):
    matrix = np.load(os.path.join(dir_data, "fire_events_evolution",filename))
    if (matrix.shape[1]>200):
        filenames.append(filename)
print("[INFO] Collected %d fire events"%len(filenames))
for filename in filenames:
    matrix = np.load(os.path.join(dir_data, "fire_events_evolution",filename))
    matrix = np.where(matrix==0, np.nan, matrix) 
    toAdd = np.array([[[0]+list(range(0,matrix.shape[2]-1))]])
    matrix = matrix+toAdd
    
    # np.nanmax(matrix[:,:,10])
    gg= np.where(np.isnan(matrix[:,:,1:]), 0, matrix[:,:,1:])
    gg = np.cumsum(gg,axis=2)
    gg= np.where(gg==0, np.nan, gg)
    matrix[:,:,1:] = gg
    
    
    center = (matrix.shape[0]/2, matrix.shape[1]/2)
    
    color = "grey"
    lw=0.5
    circle1 = plt.Circle(center, center[0], color=color,fill= False, linewidth = lw)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,3))
    ax=ax1
    ax.add_patch(circle1)
    ax.plot([0,2*center[0]],[center[0], center[0]], color = color,linewidth = lw)
    ax.plot([center[0],center[0] ],[0, 2*center[0]], color = color,linewidth = lw)
    ax.plot([center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
            [center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
                color = color,linewidth = lw)
    ax.plot([center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
            [center[0]+center[0]/np.sqrt(2),center[0]-center[0]/np.sqrt(2) ],\
                color = color,linewidth = lw)
    ax.axis('off')
    
    ax.imshow(matrix[:,:,-1], cmap = "plasma",vmin = 0, vmax = matrix.shape[2]-2)
    # ax.imshow(matrix[:,:,-1], cmap = "plasma",vmin = 0, vmax = 0)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_aspect('equal')
         
    colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
              '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
              '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
              
    cmap =  ListedColormap(sns.color_palette(colors).as_hex()) 
    ax.set_title(filename)
    ax=ax2
    
    x_axis = np.linspace(-1, 1, matrix.shape[0])[:, None]
    y_axis = np.linspace(-1, 1, matrix.shape[0])[None, :]
    arr = np.sqrt(x_axis ** 2 + y_axis ** 2)<=1
    arr = np.where(arr==True,arr,np.nan)
    ax.imshow(matrix[:,:,0]*arr,cmap = cmap,vmin = 50, vmax = 200)
    
    circle1 = plt.Circle(center, center[0], color=color,fill= False, linewidth = lw)
    ax.add_patch(circle1)
    ax.plot([0,2*center[0]],[center[0], center[0]], color = color,linewidth = lw)
    ax.plot([center[0],center[0] ],[0, 2*center[0]], color = color,linewidth = lw)
    ax.plot([center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
            [center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
                color = color,linewidth = lw)
    ax.plot([center[0]-center[0]/np.sqrt(2),center[0]+center[0]/np.sqrt(2) ],\
            [center[0]+center[0]/np.sqrt(2),center[0]-center[0]/np.sqrt(2) ],\
                color = color,linewidth = lw)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

#events_which_match_lfmc_pattern = [78493,78399,  77629,73169,77407 ]
# interesting ones = [73071
# elevation dominated: 77659
# 20% fire is random = 78139 71944

# 20% not enough LFMC = 77723 73703