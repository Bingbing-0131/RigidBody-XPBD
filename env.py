"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
#from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from model import multiple_body
import igl
from rigid_body import Rigidbody
from constraint import Hinge_Constraint

V, _, _, F, _, _ = igl.read_obj("butterfly_left_test.obj")

cor_1=[]
for f in F:
    cor_1.append(np.array([V[f[0]],V[f[1]],V[f[2]]]))  # Shape (M x 3 x 3), where M is the number of triangles
cor_1 = (cor_1 - np.array([[94.00594763,9.42199516,35.9794667]]))*np.array([0.005,0.005*8, 0.005])
num_mesh_1 = F.shape[0]

rigid_body_1 = Rigidbody(num_mesh_1)
    
rho_ = 100*np.ones(num_mesh_1)  # Assign uniform density

rigid_body_1.Initialize_geometry(cor_1, rho_)

    
V_2, _, n, F_2, _, _ = igl.read_obj("butterfly_mid_test.obj")
cor_=[]
# Alternatively, as a numpy array for all triangle coordinates
for f in F_2:
    cor_.append(np.array([V_2[f[0]],V_2[f[1]],V_2[f[2]]]))  # Shape (M x 3 x 3), where M is the number of triangles
cor_ = (cor_ - np.array([[94.00594763,9.42199516,35.9794667]]))*np.array([0.005,0.005*8, 0.005])
num_mesh = F_2.shape[0]
rigid_body_2 = Rigidbody(num_mesh)
    
rho_ = 1000*np.ones(num_mesh)  # Assign uniform density
    
rigid_body_2.Initialize_geometry(cor_, rho_)


    #V, _, n, F, _, _ = igl.read_obj("butterfly_right_test.obj")
cor_3 = cor_1 * np.array([-1.0,1.0,1.0])
rigid_body_3 = Rigidbody(num_mesh_1)
    
rho_ = 100*np.ones(num_mesh_1)  # Assign uniform density
    
rigid_body_3.Initialize_geometry(cor_3, rho_)

dt = 0.01

initial_pos_1 = np.array([2.24822454e-01,1.18554695e-10,1.65144459e-01])
initial_pos_2 = np.array([8.46356508e-12,1.18554695e-10,2.27189505e-11])
initial_pos_3 = np.array([-2.24822454e-01,1.18554695e-10,1.65144459e-01])

class FlierEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.y_threshold_max = 2.0
        self.y_v_threshold_max = 8.0
        self.z_threshold_max = np.finfo(np.float32).max
        self.z_v_threshold_max = 8.0
    
        self.y_threshold_min = -1.0
        self.y_v_threshold_min = -4.0
        self.z_threshold_min = 0.0
        self.z_v_threshold_min = -3.0

        self.model = multiple_body([rigid_body_1,rigid_body_2,rigid_body_3])
        #print(rigid_body_1.velocities)

        constraint_hinge_1=Hinge_Constraint(rigid_body_1,rigid_body_2,np.array([2.72835145e-02,0.0,9.37031485e-02])-rigid_body_1.position,np.array([2.72835145e-02,0.0,9.37031485e-02]),np.array([2.99535359e-02,0.0,-6.86618395e-02])-rigid_body_1.position,np.array([2.99535359e-02,0.0,-6.86618395e-02]))
        self.model.add_constraint(constraint_hinge_1)
        #print((np.array([[99.9960021972656250,9.4219951,22.2520008087158203]])-np.array([[94.00529501,9.42199516,35.9843687]]))*np.array([0.005,0.005*8, 0.005]))
        #constraint_hinge_2=Hinge_Constraint(rigid_body_2,rigid_body_3,np.array([-2.94742213e-02,0.0,1.01415666e-01]),np.array([-2.94742213e-02,0.0,1.01415666e-01])-rigid_body_3.position, np.array([-2.86112224e-02,0.0,-4.83662159e-02]),np.array([-2.86112224e-02,0.0,-4.83662159e-02])-rigid_body_3.position)
        constraint_hinge_2=Hinge_Constraint(rigid_body_2,rigid_body_3,np.array([-2.72835145e-02,0.0,9.37031485e-02]),np.array([-2.72835145e-02,0.0,9.37031485e-02])-rigid_body_3.position,np.array([-2.99535359e-02,0.0,-6.86618395e-02]),np.array([-2.99535359e-02,0.0,-6.86618395e-02])-rigid_body_3.position)
        self.model.add_constraint(constraint_hinge_2)
        #constraint_hinge_3=Hinge_Constraint(rigid_body_1,rigid_body_3,-rigid_body_1.position, -rigid_body_3.position,np.zeros(3),np.zeros(3))
        #self.model.add_constraint(constraint_hinge_3)
        
        

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high_state = np.array(
            [
                self.y_threshold_max * 2,
                np.finfo(np.float32).max,
                self.y_v_threshold_max * 2,
                self.z_v_threshold_max * 2,
            ],
            dtype=np.float32,
        )
        
        low_state = np.array(
            [
                self.y_threshold_min * 2,
                self.z_threshold_min * 2,
                self.y_v_threshold_min * 2,
                self.z_v_threshold_min * 2,
            ],
            dtype=np.float32,
        )
        
        self.max_action_x = 2.0
        self.max_action_y = 2.0

        max_action = np.array(
            [
                self.max_action_x,
                self.max_action_y
            ],
            dtype=np.float32,
        )

        min_action = np.array(
            [
                -self.max_action_x,
                -self.max_action_y
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.max_action_x, -self.max_action_y], dtype=np.float32),
            high=np.array([self.max_action_x, self.max_action_y], dtype=np.float32),
            dtype=np.float32,
        )


        self.observation_space = spaces.Box(low_state, high_state,dtype=np.float32)
        '''
        #self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        '''
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        self.count = 0

    def step(self, action: np.ndarray):
        self.count += 1
        torque_x = np.clip(action[0], -self.max_action_x, self.max_action_x)
        torque_y = np.clip(3*action[1], -self.max_action_y, self.max_action_y)
    
        #self.model.simulates_with_polyscope(self.count)
        self.model.UpdateDynamics(torque_x,torque_y)
        pos = np.zeros(3)
        vel = np.zeros(3)
        for body in self.model.bodies:
            pos += body.Mass*body.position/self.model.Mass
            vel += body.Mass*body.velocities/self.model.Mass
        position_x, position_y, position_z = pos
        velocity_x, velocity_y, velocity_z = vel
        #print(pos)
        norm1_x, norm1_y, norm1_z = self.model.bodies[0].norm_world
        norm2_x, norm2_y, norm2_z = self.model.bodies[1].norm_world
        norm3_x, norm3_y, norm3_z = self.model.bodies[2].norm_world

        omega1_x, omega1_y, omega1_z = self.model.bodies[0].omega_vec
        omega2_x, omega2_y, omega2_z = self.model.bodies[1].omega_vec
        omega3_x, omega3_y, omega3_z = self.model.bodies[2].omega_vec


        # Convert a possible numpy bool to a Python bool.
        terminated = (
            position_y < self.y_threshold_min or position_y > self.y_threshold_max or
            position_z < self.z_threshold_min or position_z > self.z_threshold_max or
            velocity_y < self.y_v_threshold_min or velocity_y > self.y_v_threshold_max or
            velocity_z < self.z_v_threshold_min or velocity_z > self.z_v_threshold_max or
            norm1_y <= 0 or norm2_y <=0 or norm3_y <=0
        )

        if terminated:

            if position_y < self.y_threshold_min or position_y > self.y_threshold_max:
                print("Termination triggered by position_y:", position_y)
                terminated = True

            if position_z < self.z_threshold_min or position_z > self.z_threshold_max:
                print("Termination triggered by position_z:", position_z)
                terminated = True

            if velocity_y < self.y_v_threshold_min or velocity_y > self.y_v_threshold_max:
                print("Termination triggered by velocity_y:", velocity_y)
                terminated = True

            if velocity_z < self.z_v_threshold_min or velocity_z > self.z_v_threshold_max:
                print("Termination triggered by velocity_z:", velocity_z)
                terminated = True

            if norm1_y < 0.0 or norm2_y < 0.0:
                print("Termination triggered by angle:",norm1_y,norm2_y,norm3_y)
                terminated = True

        reward = 0
        #print(action)
        if not terminated:
            reward = 1.0 - (np.abs(torque_x)+np.abs(torque_y))
        else:
            reward = 0.0
            self.count = 0

        self.state = np.array([position_y, position_z, velocity_y, velocity_z, 
                                norm1_x,norm1_y,norm1_z,
                                norm2_x, norm2_y,norm2_z,
                                norm3_x,norm3_y,norm3_z,
                                omega1_x,omega1_y,omega1_z,
                                omega2_x, omega2_y,omega2_z,
                                omega3_x,omega3_y,omega3_z,
                                ],dtype=np.float32)

        #if self.render_mode == "human":
        #    self.render()
        #print(self.state)
        return self.state, reward, terminated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        #print(angle_yaw)
        R_1 = np.eye(3)
        R_2 = np.eye(3)
        R_3 = np.eye(3)
        mean = 0.0
        std = 0.5
        L_ = 0.5#self.np_random.normal(loc=2.0, scale=1.0) 
        L_ = np.clip(L_, -2.0, 10.0)
        #print(L_)
    
        L_1 = np.array([0.0,0.0,-L_])
        L_2 = np.zeros(3)
        L_3 = np.array([0.0,0.0,L_])

        v_1 = np.array([0.0,0.0,1.0])
        v_2 = np.array([0.0,0.0,1.0])
        v_3 = np.array([0.0,0.0,1.0])

        
        initial_pos_1 = np.array([2.24822454e-01,1.18554695e-10,1.65144459e-01])
        initial_pos_2 = np.array([8.46356508e-12,1.18554695e-10,2.27189505e-11])
        initial_pos_3 = np.array([-2.24822454e-01,1.18554695e-10,1.65144459e-01])

        self.model.bodies[0].position = initial_pos_1 
        self.model.bodies[1].position = initial_pos_2 
        self.model.bodies[2].position = initial_pos_3 
        

        self.model.bodies[0].Initialize_dynamic(v_1, L_1, R_1,dt)
        self.model.bodies[1].Initialize_dynamic(v_2, L_2, R_2,dt)
        self.model.bodies[2].Initialize_dynamic(v_3, L_3, R_3,dt)


        norm_1 = self.model.bodies[0].norm_world
        norm_2 = self.model.bodies[1].norm_world
        norm_3 = self.model.bodies[2].norm_world
        
        pos = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        for body in self.model.bodies:
            pos += body.Mass*body.position
            vel += body.Mass*body.velocities
        position_x, position_y, position_z = pos/(self.model.Mass)
        velocity_x, velocity_y, velocity_z = vel/(self.model.Mass)
        # Ensure norms are properly converted to NumPy arrays
        norm1_x,norm1_y,norm1_z = np.array(norm_1, dtype=np.float32)
        norm2_x,norm2_y,norm2_z, = np.array(norm_2, dtype=np.float32)
        norm3_x,norm3_y,norm3_z = np.array(norm_3, dtype = np.float32)
        omega1_x,omega1_y,omega1_z = self.model.bodies[0].omega_vec
        omega2_x, omega2_y,omega2_z = self.model.bodies[1].omega_vec
        omega3_x,omega3_y,omega3_z = self.model.bodies[2].omega_vec


        # Stack all components into a single flat NumPy array
        self.state = np.array([position_y, position_z, velocity_y, velocity_z, 
                                norm1_x,norm1_y,norm1_z,
                                norm2_x,norm2_y,norm2_z,
                                norm3_x,norm3_y,norm3_z,
                                omega1_x,omega1_y,omega1_z,
                                omega2_x, omega2_y,omega2_z,
                                omega3_x,omega3_y,omega3_z,
                                ],dtype=np.float32)
        print('reset')
        #if self.render_mode == "human":
        #    self.render()
        return self.state

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False