# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:08:40 2017

@author: ps09og
"""

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pdb
import pandas as pd

def circle2(r = 1000, road_point_interval = 1, curve_arc_len_rads = np.pi/2):
    
    curve_arc_len_meters = curve_arc_len_rads * np.abs(r)
    print("Curve is {}m long".format(curve_arc_len_meters))
#    pdb.set_trace()
    n_road_points = 1 + curve_arc_len_meters / road_point_interval
    circle_center = np.array([0, r])
    curve_start_angle = -np.pi / 2.0
#    pdb.set_trace()
    circle_angles = np.linspace(curve_start_angle, curve_start_angle + curve_arc_len_rads, n_road_points)

    road_x = circle_center[0] + r * np.cos(circle_angles)
    road_y = circle_center[1] + r * np.sin(circle_angles)
    
    return np.array([road_x, road_y]).T

def angle_between_vectors(a, b):
    """Get the angle between vectors a and b"""
    
    return np.arccos(np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))) )

class vehicle:
    
    def __init__(self, pos, heading, speed, desired_trajectory, sight_point_distances, Kni, Knp, Kf, sigma):
        
        self.pos = np.array([pos[0], pos[1]])
        self.heading = heading #heading angle
        self.speed = speed        
        
        
        self.sigma = sigma
        
        self.sight_point_distances = sight_point_distances        
        self.sight_point_angles_old = np.array([0.0,0.0])
        self.forward_vector = np.array([np.sin(self.heading), np.cos(self.heading)])
        
        self.Kni = Kni
        self.Knp = Knp
        self.Kf = Kf
        
        self.heading_dot = 0
        
        self.pos_history = []
        self.heading_history = []
        self.heading_dot_history = []
        self.sight_point_history = []   
        self.sight_point_angle_history = []
       
        self.desired_trajectory = desired_trajectory
        
        self.get_sight_point_angles()
        self.save_history()
        
    def get_sight_point_angles(self):
        """Get the sight points"""
        
        sight_point_distances = self.sight_point_distances
        n_sight_points= len(sight_point_distances)
        
        #Get vectors from vehicle to road points
        vectors_to_road_points = np.array([self.desired_trajectory[:,0] - self.pos[0], 
                                           self.desired_trajectory[:,1] - self.pos[1]]).T
        #Get distances to road points
        road_point_distances = np.sqrt(np.sum(np.square(vectors_to_road_points), axis = 1))
        
        #Get the forward vector
        self.forward_vector = np.array([np.cos(self.heading), np.sin(self.heading)])        

        straight_ahead_distance_to_road_points = np.dot(vectors_to_road_points, self.forward_vector)
       
        road_point_distances[straight_ahead_distance_to_road_points < 0 ] = np.Infinity
                            
        
  
        self.sight_point_angles = np.empty(n_sight_points)
        sight_points = np.empty((n_sight_points, 2))
        i = 0
        
        for sight_point_d in sight_point_distances:
  
            #Get the absolute error between the distance to the road points and the sight pont distance
            v_abs_error = np.abs(road_point_distances - sight_point_d)
            v_abs_error_sorted = np.sort(v_abs_error) 
            
                
            road_points_sorted = self.desired_trajectory[np.argsort(v_abs_error)] #Road points sorted by absolute error
       
            road_point1 = road_points_sorted[0]
            road_point2 = road_points_sorted[1]
            
            abs_error1 = v_abs_error_sorted[0]
            abs_error2 = v_abs_error_sorted[1]
            
            total_abs_error = abs_error1 + abs_error2
            
            sight_point_x = (abs_error2 * road_point1[0] + abs_error1 * road_point2[0]) / total_abs_error
            sight_point_y = (abs_error2 * road_point1[1] + abs_error1 * road_point2[1]) / total_abs_error
            sight_point = np.array([sight_point_x, sight_point_y])
            sight_points[i] = sight_point                    
                        
            self.vector_to_sight_point = sight_point - self.pos        
            
            #My way of calculating angle
            theta_1 = np.arctan(self.vector_to_sight_point[1] / self.vector_to_sight_point[0])
            theta_2 = np.arctan(self.forward_vector[1] / self.forward_vector[0])            
            sp_angle = theta_1 - theta_2
            
            
            #Gustav way of calculating angle
#            longitudinal_dist_2_sp = np.dot(self.vector_to_sight_point, self.forward_vector)
#            lateral_dist_2_sp = np.dot(self.vector_to_sight_point, np.array([np.cos(self.heading + np.pi / 2), np.sin(self.heading + np.pi / 2)]))
#            sp_angle = np.arctan(lateral_dist_2_sp / longitudinal_dist_2_sp)

            self.sight_point_angles[i] = sp_angle     
            i += 1

#        self.sight_point_history.append(sight_points)
        
        return self.sight_point_angles    
     

    def update_two_point(self, dt):
        """update the position of the vehicle over timestep dt"""
        
        self.sight_point_angles = self.get_sight_point_angles()
        
        self.sight_point_angles_dot = (self.sight_point_angles - self.sight_point_angles_old) / dt
                                 
#        print(sight_point_angles_dot)
        
        self.heading_dot = (self.Kni * self.sight_point_angles[0] * dt +
                            self.Knp * self.sight_point_angles_dot[0] + 
                            self.Kf * self.sight_point_angles_dot[1] ) + np.random.normal(0, self.sigma)
        
        self.sight_point_angles_old = self.sight_point_angles
        
        
        self.heading = self.heading + self.heading_dot * dt #+ np.random.normal(0, 0.005)
        
        x_change = self.speed * dt * np.sin(self.heading)
        y_change = self.speed * dt * np.cos(self.heading)
        
        self.pos = self.pos + np.array([x_change, y_change]) #+  np.random.normal(0, self.sigma, size = 2)
        
        self.save_history()


        
    def save_history(self):
        
        self.pos_history.append(self.pos)        
        self.heading_history.append(self.heading)
        self.heading_dot_history.append(self.heading_dot)
        self.sight_point_angle_history.append(self.sight_point_angles)
    

        
        
if __name__ == '__main__':
    

    center_curve = circle2()

    plt.plot(center_curve[:,0], center_curve[:,1], '--b')
    plt.xlim([-50, 1000])
    plt.ylim([-50, 1000])
       
    #Sight point params
    speed = 13.4
    near_point_time = 0.5
    far_point_time = 2.0
    sight_point_distances = [near_point_time * speed, far_point_time * speed]

    myCar = vehicle(pos = center_curve[0] + np.array([0, 0]), heading = 0, speed = speed, 
                    desired_trajectory= center_curve, sight_point_distances = sight_point_distances , 
                    Kni = -0.1, Knp = -0.4, Kf = -0.45, 
                    sigma = 0.00001) #initalise a car object
    
    

    
    plt.scatter(myCar.pos[0], myCar.pos[1], color = 'r')
            
    myCar.get_sight_point_angles()
#    plt.scatter(myCar.sight_point_history[0][:,0], myCar.sight_point_history[0][:,1], color = 'b')
   
    plt.arrow(myCar.pos[0], myCar.pos[1], 
                      myCar.vector_to_sight_point[0] , myCar.vector_to_sight_point[1],
                               head_width=5, head_length=5, color = 'b')
    
    print(np.rad2deg(myCar.get_sight_point_angles()))
    print(myCar.forward_vector)
      
    fps = 30
    dt = 1 / fps
    run_time = 80 #seconds
    time = 0
    
    close_sp_all = []
    far_sp_all = []
    i = 0 
    while time < run_time:
        
        time += dt        
       
        i += 1
        myCar.update_two_point(dt)

    positions = np.array(myCar.pos_history)
    heading = np.array(myCar.heading_history)
    heading_dot = np.array(myCar.heading_dot_history)
    
    
    #Plot the vehicle path    
    plt.plot(positions[:,0], positions[:,1], '--r', linewidth = 3)
    
    sp_angles = np.array(myCar.sight_point_angle_history)
    sp_angles_dot = np.diff(sp_angles, axis = 0) / dt
    heading_dot = np.array(myCar.heading_dot_history)
    
    data = pd.DataFrame({'near_dot': sp_angles_dot[:,0], 'far_dot': sp_angles_dot[:,1], 'near': sp_angles[1:, 0], 'heading_dot': heading_dot[1:]})
    
    data.to_csv("SG_steering_data.csv")
    
    plt.show()
#    plt.plot(np.array(myCar.sight_point_history)[:,0,0], np.array(myCar.sight_point_history)[:,0,1], 'ro')
#    plt.plot(np.array(myCar.sight_point_history)[:,1,0], np.array(myCar.sight_point_history)[:,1,1], 'yo')

    
#    plt.show()
#    
#    plt.figure()
#    plt.plot(close_sp_all, 'r')
#    plt.plot(far_sp_all, 'b')
#    
#    
    
#    import pystan, pickle
#    
#    try:
#        model = pickle.load(open("cross_track_model2.pkl", "rb"))
#    except:
#        model = pystan.StanModel(file = open("cross_track_model_fit2.stan", 'r'))    
#        pickle.dump(model, open("cross_track_model2.pkl", 'wb'))
#    
#    
#    stan_data = dict(N = len(myCar.ct_error_history), 
#                     ct_error = myCar.ct_error_history, 
#                     heading_dot = myCar.heading_dot_history,
#                     start_pos = myCar.pos_history[0],
#                     start_heading = myCar.heading_history[0],
#                     N_steps = myCar.desired_trajectory.shape[0],
#                     curve = myCar.desired_trajectory,
#                     speed = myCar.speed,
#                     dt = dt)
#    
#
##    fit = model.sampling(data = stan_data, algorithm = 'Fixed_param', 
##                         iter = 1, chains = 1)
##    
#    fit = model.sampling(data = stan_data, 
#                         iter = 2500, chains = 4, 
#                         refresh = 50, pars = ['sigma', 'p', 'pos'])
#    
#    samps = fit.extract()
#    
#    for i in np.random.randint(0, samps['pos'].shape[0], 100):
#        plt.plot(samps['pos'][i,:,0], samps['pos'][i,:,1], 'g', alpha = 0.05)
#        
#        #print(samps['pos'][i,:,0])
#
#    plt.figure()
#    sns.distplot(samps['p'])
#    plt.axvline(myCar.p)
##
##    plt.figure()
##    plt.plot(samps['pos'][:,-1,0])
##    
##    
#    print(fit)
