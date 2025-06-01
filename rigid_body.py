import numpy as np
from aerodynamics import Mesh
#from pyquaternion import Quaternion
import math
#import Utils
from scipy.linalg import logm

def vector_to_matrix(omega):
    #print("omega:", omega, "shape:", np.shape(omega)) 
    omega_x, omega_y, omega_z = omega
    return np.array([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ])

def matrix_to_vector(matrix):
    return np.array([
        matrix[2, 1],  # omega_x = -matrix[1, 2]
        matrix[0, 2],  # omega_y = -matrix[2, 0]
        matrix[1, 0]   # omega_z = -matrix[0, 1]
    ])

import numpy as np
from pyquaternion import Quaternion

def compute_omega_from_quaternion(R_before, R_after, dt):
    q1 = Quaternion(matrix=R_before)
    q2 = Quaternion(matrix=R_after)
    #print(q1)
    #print(q2)
    
    # Compute relative quaternion
    q_rel = q2 * q1.inverse
    
    # Ensure the quaternion is in the shorter path (w >= 0)
    if q_rel.scalar < 0:
        q_rel = -q_rel
    
    # Extract the vector part (imaginary components)
    q_vec = q_rel.vector  # [x, y, z] components
    
    # Compute angular velocity using the small angle approximation
    # For small rotations: ω ≈ 2 * q_vec / dt
    # For general case: ω = 2 * atan2(||q_vec||, |q_w|) * q_vec / (||q_vec|| * dt)
    
    q_vec_norm = np.linalg.norm(q_vec)
    
    if q_vec_norm < 1e-6:
        # No rotation case
        omega = np.array([0.0, 0.0, 0.0])
    else:
        # General case: use atan2 for numerical stability
        angle = 2 * np.arctan2(q_vec_norm, abs(q_rel.scalar))
        omega = (angle / dt) * (q_vec / q_vec_norm)
    
    #print(f"Angular velocity: {omega}, Relative quaternion: {q_rel}")
    return omega


class Rigidbody:
    def __init__(self,num_mesh,index):
        self.N = num_mesh
        self.index = index
        self.meshes = [Mesh() for _ in range(num_mesh)]
        self.Mass = 0
        self.position = np.zeros(3)
        self.velocities = np.zeros(3)
        self.inertia_body = np.zeros((3,3))
        self.inertia_body_inv = np.zeros((3,3))
        self.inertia = np.zeros(3)
        self.inertia_inv = np.zeros(3)
        self.R = np.eye(3)
        self.angular_momentum = np.zeros(3)
        self.dt = 0.01
        self.omega_vec = np.zeros(3)
        self.omega_mat = np.zeros((3,3))
        self.tmp_momentum_add = np.zeros(3)
        self.tmp_angular_momentum_add = np.zeros(3)
        self.inv_mass = 0.0
        self.torque = np.zeros(3)
        self.coming_velocity = np.array([0.0,-0.0,-0.0])
        self.g = np.array([0.0,-9.8,-0.0])
        self.rho = 0.1
        self.norm_world = np.zeros(3) 
        self.norm_body = np.array([0.0,1.0,0.0])
        self.prev_pos = np.zeros(3)
        self.R_prev = np.eye(3)
        
        
    def Initialize_geometry(self, cor_, rho_):
        for i in range(self.N):
            self.meshes[i].initialize_geometry(cor_[i], rho_[i])
            
            self.Mass += self.meshes[i].mass
            self.position +=self.meshes[i].mass * self.meshes[i].x_c 
        self.position = self.position/self.Mass
        #print(self.position)

        self.inv_mass = 1.0 / self.Mass
        #print(self.Mass)
        for i in range(self.N):
            self.meshes[i].set_r(self.position)
            self.meshes[i].compute_inertia()
            self.inertia_body += self.meshes[i].inertia
        
        self.inertia_body_inv = np.linalg.inv(self.inertia_body)
        
    
    def Initialize_dynamic(self, v_, angular_, R_,dt):
        self.velocities = v_
        self.angular_momentum = angular_ * 0.00001
        self.R = R_
        
        self.inertia_inv = self.R @ self.inertia_body_inv @ self.R.T
        self.inertia = self.R @ self.inertia_body @ self.R.T
        self.omega_vec =  self.inertia_inv @ self.angular_momentum
        
        self.omega_mat = vector_to_matrix(self.omega_vec)
        for i in range(self.N):
            self.meshes[i].update(R_,self.omega_mat, self.position, self.velocities)
        self.dt = dt
        self.norm_world = R_ @ self.norm_body

    def set_coming_velocity(self,velocity):
        self.coming_velocity = velocity

    def apply_aero_force(self):
        for i in range(self.N):
            aero_force = self.meshes[i].compute_aerodynamic_force(self.coming_velocity)
            self.tmp_momentum_add +=  aero_force + self.g * self.meshes[i].mass
            self.tmp_angular_momentum_add += np.cross(self.meshes[i].r_c_world, aero_force + self.g * self.meshes[i].mass)
            #print(self.meshes[i].r_c_world, aero_force,np.cross(self.meshes[i].r_c_world, aero_force))

    def apply_torque_force(self):
        #print(self.R)
        #print(self.torque)
        self.tmp_angular_momentum_add += self.R @ self.torque

    def compute_R(self,omega, dt):
        theta = np.linalg.norm(omega * dt)
        R_increase = np.eye(3)
        if theta == 0:
            R_increase = np.eye(3)
        else:
            rot_axis = omega / np.linalg.norm(omega)
            #print(rot_axis)
            rot_axis_matrix = vector_to_matrix(rot_axis)
            R_increase = (
                np.cos(theta) * np.eye(3)
                + np.sin(theta) * rot_axis_matrix
                + (1 - np.cos(theta)) * np.outer(rot_axis, rot_axis)
            )
        return R_increase

    def advance(self, dt):
        if self.dt < dt:
            dt = self.dt
        self.prev_pos = self.position.copy()
        self.R_prev = self.R
        
        #print(self.velocities)
        self.apply_aero_force()
        #print("momentum",self.tmp_momentum_add)
        #print('momentum',self.tmp_momentum_add)
        self.apply_torque_force()
        if self.index == 1:
            self.tmp_angular_momentum_add[0] = 0.0
        #print(self.tmp_angular_momentum_add)
        a1_v = self.velocities
        a2_v = self.tmp_momentum_add / self.Mass
        #print('a2',a2_v)
        b1_v = self.velocities + dt /2 * a2_v
        a1_L = self.angular_momentum
        a2_L = self.tmp_angular_momentum_add
        b1_L = self.angular_momentum + dt / 2 * a2_L
        #print('a2_v',a2_v)
        #print('b1_v',self.velocities, b1_v)

        R_increase_tmp = self.compute_R(self.omega_vec, dt / 2)
        #print(R_increase_tmp)
        R_tmp = R_increase_tmp @ self.R
        inertia_inv_tmp = R_tmp @ self.inertia_body_inv @ R_tmp.T
        omega_tmp = inertia_inv_tmp @ b1_L
        omega_tmp_mat = vector_to_matrix(omega_tmp)
        for i in range(self.N):
            self.meshes[i].update(R_tmp, omega_tmp_mat, self.position + dt / 2 * b1_v, self.velocities + dt / 2 * a2_v)
        #print('b1',b1_v)
        

        # Intermediate state 1
        self.tmp_momentum_add = np.zeros(3)
        self.tmp_angular_momentum_add = np.zeros(3)
        self.apply_aero_force()
        self.apply_torque_force()
        #print("torque",self.tmp_angular_momentum_add)
        
        b2_v = self.tmp_momentum_add / self.Mass
        c1_v = self.velocities + dt / 2 * b2_v
        #print('c1',c1_v)
        #print(self.tmp_momentum_add)

        b2_L = self.tmp_angular_momentum_add
        c1_L = self.angular_momentum + dt / 2 * b2_L
        #print(self.angular_momentum,self.tmp_angular_momentum_add)

        R_increase_tmp = self.compute_R(self.omega_vec, dt / 2)
        R_tmp = R_increase_tmp @ self.R
        #print(R_increase_tmp)
        inertia_inv_tmp = R_tmp @ self.inertia_body_inv @ R_tmp.T
        omega_tmp = inertia_inv_tmp @ c1_L
        omega_tmp_mat = vector_to_matrix(omega_tmp)
        #print(omega_tmp_mat)
        for i in range(self.N):
            self.meshes[i].update(R_tmp, omega_tmp_mat, self.position + dt / 2 * c1_v, self.velocities + dt / 2 * b2_v)

        # Intermediate state 2
        self.tmp_momentum_add = np.zeros(3)
        self.tmp_angular_momentum_add = np.zeros(3)
        self.apply_aero_force()
        self.apply_torque_force()
        if self.index == 1:
            self.tmp_angular_momentum_add[0] = 0.0
        #print(self.index," ",self.tmp_momentum_add,self.velocities)
        c2_v = self.tmp_momentum_add / self.Mass
        d1_v = self.velocities + dt * c2_v
        #print('d1',d1_v)

        c2_L = self.tmp_angular_momentum_add
        d1_L = self.angular_momentum + dt * c2_L

        R_increase_tmp = self.compute_R(self.omega_vec, dt)
        R_tmp = R_increase_tmp @ self.R
        inertia_inv_tmp = R_tmp @ self.inertia_body_inv @ R_tmp.T
        omega_tmp = inertia_inv_tmp @ d1_L
        omega_tmp_mat = vector_to_matrix(omega_tmp)
        for i in range(self.N):
            self.meshes[i].update(R_tmp, omega_tmp_mat, self.position + dt * d1_v, self.velocities + dt * c2_v)

        # Final state
        self.tmp_momentum_add = np.zeros(3)
        self.tmp_angular_momentum_add = np.zeros(3)
        self.apply_aero_force()
        self.apply_torque_force()
        if self.index == 1:
            self.tmp_angular_momentum_add[0] = 0.0
        d2_v = self.tmp_momentum_add / self.Mass
        d2_L = self.tmp_angular_momentum_add
        #print(self.position)
        self.position += (a1_v + 2 * b1_v + 2 * c1_v + d1_v) * dt / 6
        #print(self.position)
        self.velocities += (a2_v + 2 * b2_v + 2 * c2_v + d2_v) * dt / 6
        
        R_increase_tmp = self.compute_R(self.omega_vec, dt)
        self.R = R_increase_tmp @ self.R
        self.inertia_inv = R_tmp @ self.inertia_body_inv @ R_tmp.T
        self.angular_momentum += (a2_L + 2 * b2_L + 2 * c2_L + d2_L) * dt /6
        self.omega_vec = self.inertia_inv @ self.angular_momentum

        self.omega_mat = vector_to_matrix(self.omega_vec)
        for i in range(self.N):
            self.meshes[i].update(self.R, self.omega_mat, self.position, self.velocities)

        self.tmp_momentum_add = np.zeros(3)
        self.tmp_angular_momentum_add = np.zeros(3)
        #print('L',self.omega_vec)
        #print('final',self.angular_momentum)
        #print(self.velocities)

    def PostUpdate(self,dt):
        self.velocities = (self.position - self.prev_pos)/dt
        #print(self.position)
        #sprint(self.prev_pos)
        
        R_prev = self.R_prev
        R_after = self.R
        #print(R_prev)
        #print(R_after)
        omega = compute_omega_from_quaternion(R_prev,R_after,dt)
        
        self.omega_vec = omega
        self.omega_mat = vector_to_matrix(omega)
        self.angular_momentum = self.inertia@self.omega_vec

        
        for i in range(self.N):
            self.meshes[i].update(self.R, self.omega_mat, self.position, self.velocities)
        self.norm_world = self.R @ self.norm_body
       


        