# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:26:24 2017

@author: pscog
"""

import numpy as np
import matplotlib.pylab as plt
import pdb

all_x = [] 
all_y = []

a = 1.2
b = 1.5

x = 0  
y = 0
yaw = 0

r = 0 #angular velocity
u = 50 #longitudinal velocity

v = 0.0 #latitudinal velocity
delta = np.deg2rad(100)  #steering angle (front wheel)

C_f = 0.45 #Cornering stiffness
C_r =  0.85 #Cornering stiffness (rear)
Iz = 2000   #Inertia 
m = 1000 #mass

dt = 1/30


for i in range(2000):
    
    #Calculate slip angle from lateral speed of tire
    alpha_f = (v + a * r) / u - delta
    alpha_r = (v - b * r) / u
              
    #Calculate force at each tyre using cornering stiffness     
    F_f = -C_f * alpha_f
    F_r = -C_r * alpha_r    
    
    pdb.set_trace()
    
    #Calculate the torques
    T_f = a * F_f
    T_r = -b * F_r    
    
    #Calculate acceleration (from Newtons first law)
    r_dot = (T_f + T_r) / Iz
    v_dot = (F_f + F_r)/ m - r * u        
            
    #Integrate
    r = r + r_dot * dt
    yaw = yaw + r * dt
    
    v = v * v_dot * dt
    
    x = x + np.cos(yaw) * u * dt - np.sin(yaw) * v * dt
                  
    y = y + np.sin(yaw) * u * dt - np.cos(yaw) * v * dt
                  
    all_x.append(x)
    all_y.append(y)
    