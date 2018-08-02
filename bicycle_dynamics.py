# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:07:00 2017

@author: pscog
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.stats as sts
from collections import deque
import time


class Bicycle_model:
    """A linear bicycle model"""
    
    def __init__(self, x = 0, y = 0, yaw = 0, r = 0, u = 30, v = 20, delta = 0):
        """Inputs:
            x: Initial x position
            y: Initial y position
            yaw: Initial yaw
            r: Initial angular velocity
            u: Initial longitudinal velocity
            v: Initial lateral velocity
            delta: Initial steering angle of front wheel
        """
        
        #Initial states                
        self.x = x #x position
        self.y = y #y position        
        self.yaw = yaw #yaw
        self.r = r #yaw rate (angular velocity)
        self.u = u #longitudinal velocity
        self.v = v #lateral velocity
        self.delta = delta #steering angle        
        self.rdot = 0
        self.vdot = 0
              
        #A and B are fit from the virtual sweden data set
        self.A = np.array([[-3.8986, -18.8082],
                           [0.2965, -8.7535]])
    
        self.B = np.array([2.5504, 5.0029])

    def update(self, dt, delta):        
            
        accel = np.dot(self.A, np.array([self.v, self.r])) + np.dot(self.B, delta)
        
        self.vdot = accel[0]
        self.rdot = accel[1]
        
        self.v = self.v + self.vdot * dt
        self.r = self.r + self.rdot * dt        
        self.yaw = self.yaw + self.r * dt
                
        self.x = self.x + np.cos(self.yaw) * self.u * dt - np.cos(np.pi / 2 - self.yaw) * self.v * dt                  
        self.y = self.y + np.sin(self.yaw) * self.u * dt - np.sin(np.pi / 2 - self.yaw) * self.v * dt    
       
        return self.x, self.y, self.yaw                          
 
class P_error:
    """Sevucchi and Grey get perceptual control error"""
    def __init__(self, Kni, Knp, Kf):
        
        self.old_sp_angles = None
        
        self.Kni = Kni
        self.Knp = Knp
        self.Kf = Kf
   
    
    def get_control_output(self, sp_angles, dt):
        """Pass the sight point angles and the dt and get the required change in steering angle (delta_dot)"""
        
        if self.old_sp_angles == None:
            
            self.old_sp_angles = sp_angles
            
            return 0
        
        else:               
            sp_angles_dot = (sp_angles - self.old_sp_angles) / dt
            
            
                                                 
            delta_dot = (self.Kni * sp_angles[0] * dt +
                self.Knp * sp_angles_dot[0] + 
                self.Kf * sp_angles_dot[1])
            
            self.old_sp_angles = sp_angles
            
#            print(delta_dot)
#            pdb.set_trace()
            
            if not np.isnan(delta_dot):
#                print(delta_dot)
                return delta_dot
            else:
                return 0


class intermittent_controller:
    
    def __init__(self, dt, tau_s = 0, tau_m = 0, 
                 k = 200, K = 1, 
                 sigma_n = 0, sigma_m = 0, 
                 A_threshold = 1, 
                 delta_min = 0):
        """args:
            dt: Time step of simulation 
            tau_s: Sensory delay in seconds
            tau_m: Motor delay in seconds
            k: accumulator gain
            sigma_n: Accumulator noise
            A_threshold: The accumulators threshold
            delta_min: Control refractory period - How long after an control adjustment can another be generated            
        """
            
        #Store the sensory motor delays as the number of delayed samples
        self.N_sensory_delay_samples = np.ceil(tau_s / dt)
        self.N_motor_delay_samples = np.ceil(tau_m / dt)
        self.k = k #Accumulator gain
        self.K = K #Control gain
        self.sigma_n = sigma_n #Accumulator (decision) noise
        self.sigma_m = sigma_m #Motor noise
        self.A_threshold = A_threshold        
        self.delta_min = delta_min        
        self.N_adjustments = 0 #Number of total adjustments
        self.time_since_adjustment = 0.0 #Time in seconds since the last adjustment
        
        self.g_dot = self.get_truncated_burst_rate(duration = 0.4, sigma = 2, dt = dt)
        
        self.c_dot = np.zeros(self.g_dot.shape[0] + self.N_motor_delay_samples)
        
#        pdb.set_trace()
        #Create arrays 
        self.VP_undelayed = []
        self.VP = []
        self.VC = []   
        
        self.all_A = []
        
        self.A = 0# Accumulator value
    
    def accumulate(self, perceptual_control_error, i_sample, dt):        
        """update the accumulator. Call on every frame
        args:
            perceptual_control_error: the current control error
            i_sample: the sample of the simulation"""
        
        #Add the current control error
        self.VP_undelayed.append(perceptual_control_error)       
        
        #Get the delayed perceptual control error
        if i_sample > self.N_sensory_delay_samples:            
            
            self.VP.append(self.VP_undelayed[int(i_sample - self.N_sensory_delay_samples)])       
            
        else:
            
            self.VP.append(0) 
            
        self.epsilon = self.VP[-1] - self.P_p()
        
        
        #Get the change in the accumulator
        A_change = self.gamma_GatingFcn(self.k * self.epsilon) * dt + sts.norm.rvs(0, self.sigma_n * np.sqrt(dt) )
        
        self.A = np.sign(self.A + A_change) * np.min([1.0, np.abs(self.A + A_change)])    
        
        self.all_A.append(self.A)
        
        self.time_since_adjustment = self.time_since_adjustment + dt
        
        
        #Check if a control adjustment is needed     
        self.control_adjustment()   
        
        #Now return the latest control update   
#        if self.N_adjustments > 0:
            
#            pdb.set_trace()
        next_c_dot = self.c_dot[0] #Prepare to output control adjustment
        
        self.c_dot = np.roll(self.c_dot, -1) #Shift c_dot left
        self.c_dot[-1] = 0 #Reset final value to zero
        
        return next_c_dot

       
    def control_adjustment(self):
        """Make an adjustment if the accumulator has reached threshold and 
        either the time since the last adjustment is greater than delta_min or if this is the first adjustment"""
        
        if (np.abs(self.A) >= self.A_threshold) and ((self.time_since_adjustment >= self.delta_min) or (self.N_adjustments == 0)):
#            print(self.time_since_adjustment)
#            print("Make adjustment")
            self.generate_adjustment()
            
            self.N_adjustments = self.N_adjustments + 1            
            self.reset_A() #Reset the accumulator
            
    def generate_adjustment(self):
        """Generate control adjustment and then superposition onto ongoing adjustments """
        
        #No noise
        g_i = self.epsilon * self.K
        
        #Signal dependent noise (when sigma_m == 0, g_title_i == g_tidle)
        g_tidle_i = self.K *  sts.norm.rvs(1, self.sigma_m) * self.epsilon  #Why is the gaussian centered at 1?  
        
        self.c_dot[self.N_motor_delay_samples:] = self.c_dot[self.N_motor_delay_samples:] + self.g_dot * g_tidle_i
        
        
    def reset_A(self):
        """Reset the accumulator and reset the time since the last adjustment"""
        
        self.A = 0
        self.time_since_adjustment = 0.0
            
    def P_p(self):
        """Predict the control error"""
        
        return 0
    
    def gamma_GatingFcn(self, epsilon, epsilon_0 = 0):
        
#        pdb.set_trace()
        return np.sign(epsilon) * np.max([0, np.abs(epsilon) - epsilon_0])
    
    def get_truncated_burst_rate(self, duration , sigma, dt):

        x = np.arange(0, duration + dt, dt)
        
        gaussian_std = (duration / 2) / sigma
        
        burst_rate = sts.norm.pdf(x, duration/ 2, gaussian_std)
        burst_rate = burst_rate - burst_rate[0] #Force the function to start at zero
        
        area_of_truncated_gaussian = np.trapz(burst_rate, x)
        
        burst_rate = burst_rate / area_of_truncated_gaussian #Force the function to integrate to 1
            
        assert(np.isclose(np.trapz(burst_rate, x), 1)) #Make sure this integrates to 1
        
    #    print("Burst rate integrates to: {}".format(np.trapz(burst_rate, x)))
        
    #    plt.plot(x, burst_rate)
    #    
    #    plt.show()
        
        return burst_rate
        
        

    

class Perception:
    """A driver class. Gets perceptual information and transforms it into a motor output"""
    
    def __init__(self, desired_trajectory, sight_point_distances):
        """
        args:
            desired_trajectory: The planned trajectory (e.g. road center)
            sight_point_distances: The distance ahead of the car to get the sight points
            Kni: Near integration term (from Sevuchi and Grey)
            Knp: Near 
            Kf:            
        """
          
        self.sight_point_distances = sight_point_distances                  
        self.desired_trajectory = desired_trajectory
              
        
    def get_sight_point_angles(self, x, y, yaw):
        """Get the sight point angles given the current position and yaw angle
        args:
            x: current x position
            y: current y position
            yaw: current yaw (heading angle)        
        """
        
        pos = np.array([x, y]) #Current car position
        
        sight_point_distances = self.sight_point_distances
        n_sight_points= len(sight_point_distances)
        
        
        #Get vectors from vehicle to road points
        vectors_to_road_points = np.array([self.desired_trajectory[:,0] - pos[0], 
                                           self.desired_trajectory[:,1] - pos[1]]).T
        #Get distances to road points
        road_point_distances = np.sqrt(np.sum(np.square(vectors_to_road_points), axis = 1))
        
        #Get the forward vector
        self.forward_vector = np.array([np.cos(yaw), np.sin(yaw)])       
        
        #Check which road points are behind the driver (nice little trick :  a * b = |a| * |b| * cos(theta), where theta is the angle between the two vectors
        straight_ahead_distance_to_road_points = np.dot(vectors_to_road_points, self.forward_vector)       
        road_point_distances[straight_ahead_distance_to_road_points < 0 ] = np.Infinity       
      
        self.sight_point_angles = np.empty(n_sight_points)
        sight_points = np.empty((n_sight_points, 2))
        i = 0
        
        for sight_point_d in sight_point_distances:
      
            #Get the absolute error between the distance to the road points and the sight point distance
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
                        
            self.vector_to_sight_point = sight_point - (x, y)        
            
            #My way of calculating angle
            theta_1 = np.arctan(self.vector_to_sight_point[1] / self.vector_to_sight_point[0])
            theta_2 = np.arctan(self.forward_vector[1] / self.forward_vector[0])            
            sp_angle = theta_1 - theta_2            
            
            self.sight_point_angles[i] = sp_angle     
            i += 1  

        return self.sight_point_angles    



def gamma_GatingFcn(epsilon, epsilon_0 = 0):
    
    return np.sign(epsilon) * np.max(0, np.abs(epsilon) - epsilon_0)



def generate_road(r = 1000, road_point_interval = .5, curve_arc_len_rads = np.pi/2):
    
    curve_arc_len_meters = curve_arc_len_rads * r
    print("Curve is {}m long".format(curve_arc_len_meters))
    
    n_road_points = 1 + curve_arc_len_meters / road_point_interval
    circle_center = np.array([0, r])
    
    curve_start_angle = -np.pi / 2.0    
    circle_angles = np.linspace(curve_start_angle, curve_start_angle + curve_arc_len_rads, n_road_points)
    
    road_x = circle_center[0] + r * np.cos(circle_angles)
    road_y = circle_center[1] + r * np.sin(circle_angles)
    
    return np.array([road_x, road_y]).T


def generate_sin_road():
    
    road_x = np.linspace(0, 5000,1000)
    road_y = 80* np.sin(np.linspace(0, 2000,1000) * 0.005)
    
    return np.array([road_x, road_y]).T

if __name__ == '__main__':

    t0 = time.time()
#    road = generate_road()
    road = generate_sin_road()
    plt.figure()
    plt.plot(road[:,0], road[:,1])    
             
    car_state = []
    fps = 100
    dt = 1 / fps
    
    sim_time = 80 #seconds
    
    iters = sim_time / dt #Number of iterations required
    
    t = 0
    
    longitudinal_speed = 30.0 
    near_point_time = 0.25
    far_point_time = 2
    
    sight_point_distances = [near_point_time * longitudinal_speed, far_point_time * longitudinal_speed]
    
    delta = 0
    vis_input = Perception(road, sight_point_distances) #Initialsise perceptual error quantity
    Car = Bicycle_model(0,0, 0, 0, longitudinal_speed, 0, delta) #Initialise car position
    P = P_error(Kni = 0.2, Knp = 0.3, Kf = 1.6)
    IT_control = intermittent_controller(dt = dt, tau_s = 0.2, k = 80, delta_min = 2)
    
    all_delta = []
    
    for i in range(int(iters)):
        
        sp = vis_input.get_sight_point_angles(Car.x, Car.y, Car.yaw) #Get the sight point from the cars current position
        
        delta_dot = P.get_control_output(sp, dt)
        
        delta_dot2 = IT_control.accumulate(perceptual_control_error= delta_dot, i_sample = i, dt = dt)
        
        all_delta.append(delta_dot2)
        
        #        delta = delta + delta_dot * dt  
        
        delta = delta + delta_dot2 * dt
        
        
        #Calculate lateral and angular acceleration
        max_steering_angle = np.pi / 2
        
        if (delta > max_steering_angle) or (delta < -max_steering_angle):
            delta = max_steering_angle * np.sign(delta) 
#        print(np.rad2deg(delta_dot), np.rad2deg(delta_dot2))
        
#        print(delta_dot, delta)
           
        out = Car.update(dt, delta) #Euler physics step
        car_state.append(out)        
    

       
    t1 = time.time()
    
    print("That took {} sec".format(t1 -t0))
        
    x = [c[0] for c in car_state]    
    y = [c[1] for c in car_state]    
    yaw = [c[2] for c in car_state]  
    plt.plot(x, y, 'r--')       
    plt.figure()
    plt.plot(y, 'r--')  
    
    plt.figure()
    #plt.plot(all_delta)
    plt.plot(dt * (np.arange(len(IT_control.all_A)) +1), IT_control.all_A, 'k-')
    #plt.xlim([-220, 220])
    #plt.ylim([-20, 440])
    plt.show()



