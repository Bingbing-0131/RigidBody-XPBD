import numpy as np
import math
from rigid_body import Rigidbody

def vector_to_matrix(omega):
    omega_x, omega_y, omega_z = omega
    return np.array([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ])

def compute_R(omega):
    theta = np.linalg.norm(omega)
    R_increase = np.eye(3)
    if theta == 0:
        R_increase = np.eye(3)
    else:
        rot_axis = omega / np.linalg.norm(omega)
        rot_axis_matrix = vector_to_matrix(rot_axis)
        R_increase = (
            np.cos(theta) * np.eye(3)
            + np.sin(theta) * rot_axis_matrix
            + (1 - np.cos(theta)) * np.outer(rot_axis, rot_axis)
        )
    return R_increase

def matrix_to_vector(matrix):
    """
    Convert a skew-symmetric matrix to a vector.

    Args:
        matrix (np.ndarray): A 3x3 skew-symmetric matrix.

    Returns:
        np.ndarray: A 3D vector corresponding to the input matrix.
    """
    return np.array([
        matrix[2, 1],  # omega_x = -matrix[1, 2]
        matrix[0, 2],  # omega_y = -matrix[2, 0]
        matrix[1, 0]   # omega_z = -matrix[0, 1]
    ])


class Hinge_Constraint():
    def __init__(self, body1, body2, r_1, r_2, r_3,r_4, stiffness=float('inf'), damping = float('inf')):
        self.body1 = body1
        self.body2 = body2
        self.lambda_prev = 0.0
        self.stiffness = stiffness
        self.beta = damping
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
        self.r_4 = r_4
        self.angle = 0.0
        self.omega = 0.0
        self.angle = 0.0
        #self.direction = direction_

    '''
    def SetAngularConstraint(self,dt):
        angle_sin = np.cross(self.body1.R @ self.direction, self.body2.R @ self.direction)
        angle_sin_norm = np.linalg.norm(angle_sin)
        print(angle_sin)
        if angle_sin_norm == 0:
            return
        normal_vector = angle_sin / angle_sin_norm

        R_increase_1 = np.eye(3)
        R_increase_2 = np.eye(3)

        w_1 = 0.0
        w_2 = 0.0

        if self.body1.inv_mass == 0.0:
            w_1 += 0.0
        else:
            w_1 += self.body1.inv_mass + normal_vector.T @ self.body1.inertia_inv @ normal_vector
        if self.body2.inv_mass == 0.0:
            w_2 += 0.0
        else:
            w_2 += self.body2.inv_mass + normal_vector.T @ self.body2.inertia_inv @ normal_vector

        delta_lambda = 0.0
        if self.stiffness != 'inf':
            delta_lambda = - angle_sin_norm / (w_1 + w_2 + 1.0 / (self.stiffness * dt * dt))
        else:
            delta_lambda = - angle_sin_norm / (w_1 + w_2)
        
        p = delta_lambda * normal_vector

        if self.body1.inv_mass == 0.0:
            R_increase_1 = np.eye(3)
        else:
            tmp_1 = -self.body1.inertia_inv @ p
            tmp_1_norm=np.linalg.norm(tmp_1)
            if tmp_1_norm==0.0:
                R_increase_1=np.eye(3)
            else:
                rot_axis_1 = tmp_1/np.linalg.norm(tmp_1)
                theta_1 = delta_lambda*tmp_1_norm
                rot_axis_matrix_1 = vector_to_matrix(rot_axis_1)
            
                R_increase_1 = (
                    np.cos(theta_1) * np.eye(3)
                    + np.sin(theta_1) * rot_axis_matrix_1
                    + (1 - np.cos(theta_1)) * np.outer(rot_axis_1, rot_axis_1)
                )
        if self.body2.inv_mass == 0.0:
            R_increase_2 = np.eye(3)
        else:
            tmp_2 = self.body2.inertia_inv @ p
            tmp_2_norm=np.linalg.norm(tmp_2)
            if tmp_2_norm==0.0:
                R_increase_2=np.eye(3)
            else:
                rot_axis_2 = tmp_2/np.linalg.norm(tmp_2)
                theta_2 = delta_lambda*tmp_2_norm
                rot_axis_matrix_2 = vector_to_matrix(rot_axis_2)
            
                R_increase_2 = (
                    np.cos(theta_2) * np.eye(3)
                    + np.sin(theta_2) * rot_axis_matrix_2
                    + (1 - np.cos(theta_2)) * np.outer(rot_axis_2, rot_axis_2)
                )


        self.body1.R = R_increase_1 @ self.body1.R
        self.body2.R = R_increase_2 @ self.body2.R

        self.body1.omega_vec+=self.body1.inertia_inv@ (-p)
        self.body2.omega_vec+=self.body2.inertia_inv@ (p)

        self.body1.omega_mat=vector_to_matrix(self.body1.omega_vec)
        self.body2.omega_mat=vector_to_matrix(self.body2.omega_vec)

        self.body1.inertia_inv=self.body1.R@self.body1.inertia_inv@self.body1.R.T
        self.body2.inertia_inv=self.body2.R@self.body2.inertia_inv@self.body2.R.T

        angle_sin = np.cross(self.body1.R @ self.direction, self.body2.R @ self.direction)
        angle_sin_norm = np.linalg.norm(angle_sin)
        print(angle_sin)
    '''
    def SetPositionConstraint_new(self,dt):
        r_rot_1=self.body1.R@self.r_3
        r_rot_2=self.body2.R@self.r_4

        pos_1=self.body1.position+r_rot_1
        pos_2=self.body2.position+r_rot_2

        d=np.linalg.norm(pos_2-pos_1)
        #print(self.r_3,self.r_4)
        ##print(r_rot_1,r_rot_2)
        ##print(pos_1,pos_2)
        #print(self.body1.position,self.body2.position)
        #print(d)
        C=d
    
        if C<=0.0:
            return
        #print('d1_',d)
        x_n=(pos_2-pos_1)/d
        
        
        tmp_1=np.cross(r_rot_1,x_n)
        tmp_2=np.cross(r_rot_2,x_n)
        
        w_1=0.0
        w_2=0.0
        
        if self.body1.inv_mass==0.0:
            w_1+=0.0
        else:
            w_1+=self.body1.inv_mass+tmp_1.T@self.body1.inertia_inv@tmp_1
            
        if self.body2.inv_mass==0.0:
            w_2+=0.0
        else:
            w_2+=self.body2.inv_mass+tmp_2.T@self.body2.inertia_inv@tmp_2
            
        if self.stiffness!='inf':
            delta_lambda=-C/(w_1+w_2+1.0/(self.stiffness*dt*dt))
        else:
            delta_lambda=-C/(w_1+w_2)
        
        self.lambda_prev+=delta_lambda
        tmp=x_n*delta_lambda

        self.body1.position-=self.body1.inv_mass*tmp
        self.body2.position+=self.body2.inv_mass*tmp
        #print(self.body1.position)

        R_increase_1=np.eye(3)
        R_increase_2=np.eye(3)
        if self.body1.inv_mass==0.0: 
            R_increase_1=np.eye(3)
        else:
            tmp_1=-self.body1.inertia_inv @ np.cross(r_rot_1,x_n)
            tmp_1_norm=np.linalg.norm(tmp_1)
            if tmp_1_norm==0.0:
                R_increase_1=np.eye(3)
            else:
                rot_axis_1 = tmp_1/np.linalg.norm(tmp_1)
                theta_1 = delta_lambda*tmp_1_norm
                rot_axis_matrix_1 = vector_to_matrix(rot_axis_1)
            
                R_increase_1 = (
                    np.cos(theta_1) * np.eye(3)
                    + np.sin(theta_1) * rot_axis_matrix_1
                    + (1 - np.cos(theta_1)) * np.outer(rot_axis_1, rot_axis_1)
                )

        if self.body2.inv_mass==0.0: 
            R_increase_2=np.eye(3)
        else:
            tmp_2=self.body2.inertia_inv @ np.cross(r_rot_2,x_n)
            tmp_2_norm=np.linalg.norm(tmp_2)
            if tmp_2_norm==0.0:
                R_increase_2=np.eye(3)
            else:
                rot_axis_2 = tmp_2/np.linalg.norm(tmp_2)
                theta_2 = delta_lambda*tmp_2_norm
                rot_axis_matrix_2 = vector_to_matrix(rot_axis_2)
            
                R_increase_2 = (
                    np.cos(theta_2) * np.eye(3)
                    + np.sin(theta_2) * rot_axis_matrix_2
                    + (1 - np.cos(theta_2)) * np.outer(rot_axis_2, rot_axis_2)
                )
    
        self.body1.R = R_increase_1 @ self.body1.R
        self.body2.R = R_increase_2 @ self.body2.R
        
        self.body1.inertia_inv=self.body1.R@self.body1.inertia_body_inv@self.body1.R.T
        self.body2.inertia_inv=self.body2.R@self.body2.inertia_body_inv@self.body2.R.T
        

        #self.body1.omega_vec+=self.body1.inertia_inv@ (np.cross(r_rot_1,-tmp/dt))
        #self.body2.omega_vec+=self.body2.inertia_inv@ (np.cross(r_rot_2,tmp/dt))

        ##self.body1.omega_mat=vector_to_matrix(self.body1.omega_vec)
        #self.body2.omega_mat=vector_to_matrix(self.body2.omega_vec)

    
        #self.body1.velocities-=self.body1.inv_mass*tmp/dt
        #self.body2.velocities+=self.body2.inv_mass*tmp/dt

        r_rot_1=self.body1.R@self.r_3
        r_rot_2=self.body2.R@self.r_4

        pos_1=self.body1.position+r_rot_1
        pos_2=self.body2.position+r_rot_2

        d=np.linalg.norm(pos_2-pos_1)
        #print('d2_',d)
        #print(self.body1.inv_mass)
        #if d>1e-5:
        #print('final',d)

    def SetPositionConstraint(self,dt):
        r_rot_1=self.body1.R@self.r_1
        r_rot_2=self.body2.R@self.r_2

        pos_1=self.body1.position+r_rot_1
        pos_2=self.body2.position+r_rot_2
        #print(self.r_1,self.r_2)
        #print(r_rot_1,r_rot_2)
        d=np.linalg.norm(pos_2-pos_1)
        #print(d)
        C=d
    
        if C<=0.0:
            return
        #print('d1',d)
        x_n=(pos_2-pos_1)/d
        
        
        tmp_1=np.cross(r_rot_1,x_n)
        tmp_2=np.cross(r_rot_2,x_n)
        
        w_1=0.0
        w_2=0.0
        
        if self.body1.inv_mass==0.0:
            w_1+=0.0
        else:
            w_1+=self.body1.inv_mass+tmp_1.T@self.body1.inertia_inv@tmp_1
            
        if self.body2.inv_mass==0.0:
            w_2+=0.0
        else:
            w_2+=self.body2.inv_mass+tmp_2.T@self.body2.inertia_inv@tmp_2
            
        if self.stiffness!='inf':
            delta_lambda=-C/(w_1+w_2+1.0/(self.stiffness*dt*dt))
        else:
            delta_lambda=-C/(w_1+w_2)
        
        self.lambda_prev+=delta_lambda
        tmp=x_n*delta_lambda

        self.body1.position-=self.body1.inv_mass*tmp
        self.body2.position+=self.body2.inv_mass*tmp
        #print(self.body1.position)

        R_increase_1=np.eye(3)
        R_increase_2=np.eye(3)
        if self.body1.inv_mass==0.0: 
            R_increase_1=np.eye(3)
        else:
            tmp_1=-self.body1.inertia_inv @ np.cross(r_rot_1,x_n)
            tmp_1_norm=np.linalg.norm(tmp_1)
            if tmp_1_norm==0.0:
                R_increase_1=np.eye(3)
            else:
                rot_axis_1 = tmp_1/np.linalg.norm(tmp_1)
                theta_1 = delta_lambda*tmp_1_norm
                rot_axis_matrix_1 = vector_to_matrix(rot_axis_1)
            
                R_increase_1 = (
                    np.cos(theta_1) * np.eye(3)
                    + np.sin(theta_1) * rot_axis_matrix_1
                    + (1 - np.cos(theta_1)) * np.outer(rot_axis_1, rot_axis_1)
                )

        if self.body2.inv_mass==0.0: 
            R_increase_2=np.eye(3)
        else:
            tmp_2=self.body2.inertia_inv @ np.cross(r_rot_2,x_n)
            tmp_2_norm=np.linalg.norm(tmp_2)
            if tmp_2_norm==0.0:
                R_increase_2=np.eye(3)
            else:
                rot_axis_2 = tmp_2/np.linalg.norm(tmp_2)
                theta_2 = delta_lambda*tmp_2_norm
                rot_axis_matrix_2 = vector_to_matrix(rot_axis_2)
            
                R_increase_2 = (
                    np.cos(theta_2) * np.eye(3)
                    + np.sin(theta_2) * rot_axis_matrix_2
                    + (1 - np.cos(theta_2)) * np.outer(rot_axis_2, rot_axis_2)
                )
    
        self.body1.R = R_increase_1 @ self.body1.R
        self.body2.R = R_increase_2 @ self.body2.R
        
        self.body1.inertia_inv=self.body1.R@self.body1.inertia_body_inv@self.body1.R.T
        self.body2.inertia_inv=self.body2.R@self.body2.inertia_body_inv@self.body2.R.T
        

        r_rot_1=self.body1.R@self.r_1
        r_rot_2=self.body2.R@self.r_2

        pos_1=self.body1.position+r_rot_1
        pos_2=self.body2.position+r_rot_2

        d=np.linalg.norm(pos_2-pos_1)