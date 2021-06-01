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
    if matrix.shape[1]>200:
        filenames.append(filename)
        
filename = filenames[2]
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

fig, ax = plt.subplots()
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

image = ax.imshow(np.empty((matrix.shape[0],matrix.shape[1])), cmap = "plasma",vmin = 0, vmax = matrix.shape[2]-2)
ax.imshow(matrix[:,:,-1], cmap = "plasma",vmin = 0, vmax = matrix.shape[2]-2)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

ax.set_ylim(0,matrix.shape[1])
ax.set_xlim(0, matrix.shape[0])
ax.set_aspect('equal')


time_template = 'T = %d'
time_text = ax.set_title('')        

# fig.canvas.mpl_connect('key_press_event', on_press)
# anim = animation.FuncAnimation(fig, update_plot, frames=update_time,
#                          interval=500, repeat=True)
# anim.running = True
# anim.direction = +1
plt.show()

colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
          '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
          '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
          
cmap =  ListedColormap(sns.color_palette(colors).as_hex()) 
fig, ax = plt.subplots()
ax.imshow(matrix[:,:,0],cmap = cmap,vmin = 50, vmax = 200)

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
