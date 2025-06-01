import numpy as np
import math

class Mesh:
    def __init__(self):  # Corrected the typo here
        self.x = np.zeros((3, 3))  # World coordinate: three vertices
        self.x_c = np.zeros(3)  # World coordinate: mass center
        self.rho = 0.0  # Density
        self.mass = 0.0  # Mass
        self.r_c_body = np.zeros(3)  # Body coordinate: the mass center according to the rigid mass center
        self.r_c_world = np.zeros(3)  # World coordinates: mass center according to the rigid body mass center
        self.r_body = np.zeros((3, 3))  # Initialize r_body
        self.v = np.zeros(3)  # Center velocity
        self.normal_body = np.zeros(3)
        self.normal_world = np.zeros(3)
        self.rho_f = 1.226
        self.inertia = np.eye(3)
        self.area = 0.0

    def compute_mass(self):
        A, B, C = self.x[0], self.x[1], self.x[2]
        AB = B - A
        AC = C - A
        cross_product = np.cross(AB, AC)
        self.area = 0.5 * np.linalg.norm(cross_product)
        self.mass = self.rho * self.area * 0.01
        self.normal_body = np.array([0.0, 1.0, 0.0])  # cross_product / np.linalg.norm(cross_product)

    def compute_inertia(self):
        x1, y1, z1 = self.r_body[0]
        x2, y2, z2 = self.r_body[1]
        x3, y3, z3 = self.r_body[2]
        mass = self.mass
        Ixx = mass * (y1**2 + y2**2 + y3**2 + y1*y2 + y1*y3 + y2*y3 + z1**2 + z2**2 + z3**2 + z1*z2 + z1*z3 + z2*z3) / 6.0
        Iyy = mass * (x1**2 + x2**2 + x3**2 + x1*x2 + x1*x3 + x2*x3 + z1**2 + z2**2 + z3**2 + z1*z2 + z1*z3 + z2*z3) / 6.0
        Izz = mass * (y1**2 + y2**2 + y3**2 + y1*y2 + y1*y3 + y2*y3 + x1**2 + x2**2 + x3**2 + x1*x2 + x1*x3 + x2*x3) / 6.0
        Ixy = -mass * (2*x1*y1 + 2*x2*y2 + 2*x3*y3 + x1*y2 + x2*y1 + x1*y3 + x3*y1 + x2*y3 + x3*y2) / 12.0
        Izx = -mass * (2*x1*z1 + 2*x2*z2 + 2*x3*z3 + x1*z2 + x2*z1 + x1*z3 + x3*z1 + x2*z3 + x3*z2) / 12.0
        Iyz = -mass * (2*y1*z1 + 2*y2*z2 + 2*y3*z3 + y1*z2 + y2*z1 + y1*z3 + y3*z1 + y2*z3 + y3*z2) / 12.0
        self.inertia = np.array([
            [Ixx, Ixy, Izx],
            [Ixy, Iyy, Iyz],
            [Izx, Iyz, Izz]
        ])
        #print(Ixx)

    def initialize_geometry(self, cor_, rho_):
        self.rho = rho_
        self.x = cor_
        self.compute_mass()
        self.x_c = (self.x[0] + self.x[1] + self.x[2]) / 3

    def set_r(self, X_c):
        self.r_c_body = self.x_c - X_c
        for i in range(3):
            self.r_body[i] = self.x[i] - X_c

    def update(self, R_, Omega, X_c, v_c):
        self.r_c_world = R_ @ self.r_c_body
        self.x_c = X_c + self.r_c_world
        self.v = v_c + Omega @ self.r_c_world
        #print(Omega,self.v)
        self.normal_world = R_ @ self.normal_body
        for i in range(3):
            self.x[i] = X_c + R_ @ self.r_body[i]
        #print(X_c,v_c,self.v)

    def compute_aerodynamic_force(self, coming_velocity):
        #self.relevant = coming_velocity - self.v
        #norm_velocity = np.dot(self.relevant, self.normal_world)
        #tangent_velocity = self.relevant - norm_velocity * self.normal_world
        force = 2*1.26 * self.area * self.normal_world * np.dot(self.normal_world, (coming_velocity-self.v))*np.linalg.norm(coming_velocity-self.v)
        #print('force',force,self.normal_world,self.v)
        return force