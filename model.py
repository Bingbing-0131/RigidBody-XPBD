import numpy as np
from rigid_body import Rigidbody
import gym
from gym import spaces
import math
import igl
from constraint import Hinge_Constraint
import matplotlib.cm as cm 
import polyscope as ps

class multiple_body:
    def __init__(self, bodies):
        self.num_body = len(bodies)
        self.bodies = bodies
        self.num_constraint = 0
        self.constraints = []
        self.it = 10
        self.dt = 0.01
        self.Mass = 0.0
        for body in self.bodies:
            self.Mass += body.Mass
        #ps.init()

    def add_body(self, rigid_body):
        self.bodies.append(rigid_body)
        self.num_body += 1

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        self.num_constraint += 1

    def initialize_angle(self):
        for i,constraint in enumerate(self.constraints):
            norm_1 = constraint.body1.norm_world
            norm_2 = constraint.body2.norm_world
            ang = np.dot(norm_1,norm_2)
            if ang > 1.0:
                ang = 1.0
            constraint.angle = math.acos(ang)
        return self.constraints[0].angle

    def initialize_coming_velocity(self,velocity):
        for i in range(self.num_body):
            self.bodies.set_coming_velocity(velocity[i])
    
    def UpdateDynamics(self,torque_x,torque_y):
        self.constraints[0].body1.torque += np.array([0.0,0.0,torque_y])
        self.constraints[0].body2.torque += np.array([torque_x,0.0,-torque_y])
        self.constraints[1].body1.torque += np.array([0.0,0.0,torque_y])
        self.constraints[1].body2.torque += np.array([0.0,0.0,-torque_y])
        for i,body in enumerate(self.bodies):
            #print(body.torque)
            body.advance(self.dt)
        
        for i in range(self.it):
            self.constraints[0].SetPositionConstraint(self.dt)
            self.constraints[1].SetPositionConstraint(self.dt)
            self.constraints[0].SetPositionConstraint_new(self.dt)
            self.constraints[1].SetPositionConstraint_new(self.dt)
            #self.constraints[2].SetPositionConstraint(self.dt)
        
        for body in self.bodies:
            body.PostUpdate(self.dt)
            body.torque=np.zeros(3)
            
    '''
    def simulate_with_polyscope(self,count):
        ps.set_up_dir("x_up")
        ps.set_ground_plane_mode("none")

        num_bodies = len(self.bodies)
        colormap = cm.get_cmap("tab10", num_bodies)
        colors = [colormap(i)[:3] for i in range(num_bodies)]

        for body_idx, body in enumerate(self.bodies):
            vertices = [v.copy() for mesh in body.meshes for v in mesh.x]
            faces = []  # Generate dummy faces for edges
            for i in range(len(vertices)):
                faces.append([i, (i + 1) % len(vertices)])  # Simple edge connections

            mesh = ps.register_surface_mesh(f"Body {body_idx + 1} Mesh", np.array(vertices), np.array(faces))
            mesh.set_color(colors[body_idx])

        def callback():
            for body_idx, body in enumerate(self.bodies):
                vertices = [v.copy() for mesh in body.meshes for v in mesh.x]
                ps.get_surface_mesh(f"Body {body_idx + 1} Mesh").update_vertex_positions(np.array(vertices))
             
        ps.set_user_callback(callback)
        ps.screenshot('tmp/' + str(count) + '.jpg')
    
    '''  
    def simulate_with_polyscope(self, steps):
        import polyscope as ps
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_front_dir('y_front')
        ps.set_ground_plane_mode("none")

        num_bodies = len(self.bodies)
        colormap = cm.get_cmap("tab10", num_bodies)
        colors = [colormap(i)[:3] for i in range(num_bodies)]

        for body_idx, body in enumerate(self.bodies):
            vertices = [v.copy() for mesh in body.meshes for v in mesh.x]
            faces = []  # Generate dummy faces for edges
            for i in range(len(vertices)):
                faces.append([i, (i + 1) % len(vertices)])  # Simple edge connections

            mesh = ps.register_surface_mesh(f"Body {body_idx + 1} Mesh", np.array(vertices), np.array(faces))
            mesh.set_color(colors[body_idx])

        current_step = {"step": 0}

        def callback():
            step = current_step["step"]
            if step < steps:
                t = step * self.dt
                torque = -0.01#np.array([0.0,10.0, 0.0])
                self.UpdateDynamics(torque,0.0)


                for body_idx, body in enumerate(self.bodies):
                    vertices = [v.copy() for mesh in body.meshes for v in mesh.x]
                    ps.get_surface_mesh(f"Body {body_idx + 1} Mesh").update_vertex_positions(np.array(vertices))
                current_step["step"] += 1
            
        ps.set_user_callback(callback)
        ps.show()
    



def main():
    dt = 0.01
    # Load plate.obj
    V, _, _, F, _, _ = igl.read_obj("butterfly_left_test.obj")

    cor_1=[]
    for f in F:
        cor_1.append(np.array([V[f[0]],V[f[1]],V[f[2]]]))  # Shape (M x 3 x 3), where M is the number of triangles
    cor_1 = (cor_1 - np.array([[94.00594763,9.42199516,35.9794667]]))*np.array([0.005,0.005*8, 0.005])
    num_mesh_1 = F.shape[0]

    rigid_body_1 = Rigidbody(num_mesh_1,0)
    
    rho_ = 1*np.ones(num_mesh_1)  # Assign uniform density

    rigid_body_1.Initialize_geometry(cor_1, rho_)
    v_1 = np.array([0.0, 0.0, 0.0])
    theta = math.pi/3
    #L_ = np.array([0.0 , -8.0*np.sin(theta),  8.0*np.cos(theta)])
    L_=np.array([0.0, 0.0, -10.0])
    
    R_ = np.array([
        [1.0, 0.0, 0.0],
        [0.0,np.sin(theta),np.cos(theta)],
        [0.0,-np.cos(theta),np.sin(theta)]
    ])
    
    R_=np.eye(3)
    rigid_body_1.Initialize_dynamic(v_1, L_, R_,dt)
    print(rigid_body_1.position)
    
    V_2, _, n, F_2, _, _ = igl.read_obj("butterfly_mid_test.obj")
    cor_=[]
    # Alternatively, as a numpy array for all triangle coordinates
    for f in F_2:
        cor_.append(np.array([V_2[f[0]],V_2[f[1]],V_2[f[2]]]))  # Shape (M x 3 x 3), where M is the number of triangles
    cor_ = (cor_ - np.array([[94.00594763,9.42199516,35.9794667]]))*np.array([0.005,0.005*8, 0.005])
    num_mesh = F_2.shape[0]
    rigid_body_2 = Rigidbody(num_mesh,1)
    print(rigid_body_2.position)
    rho_ = 10*np.ones(num_mesh)  # Assign uniform density
    
    rigid_body_2.Initialize_geometry(cor_, rho_)
    print(rigid_body_2.position)

    v_2 = np.array([0.0, 0.0, 0.0])
    L_2 = np.array([0.0, 0.0, 0.0])
    #L=np.zeros(3)
    R_ = np.eye(3)
    
    R_2 = np.array([
        [1.0, 0.0, 0.0],
        [0.0,np.sin(theta),np.cos(theta)],
        [0.0,-np.cos(theta),np.sin(theta)]
    ])
    
    
    R_2=np.eye(3)
    rigid_body_2.Initialize_dynamic(v_2, L_2, R_2,dt)


    #V, _, n, F, _, _ = igl.read_obj("butterfly_right_test.obj")
    cor_3 = cor_1 * np.array([-1.0,1.0,1.0])
    # Alternatively, as a numpy array for all triangle coordinates
    #for f in F:
    #    cor_.append(np.array([-V[f[0]],-V[f[1]],-V[f[2]]]))  # Shape (M x 3 x 3), where M is the number of triangles
    #cor_ = (cor_ - np.array([[94.00529501,9.42199516,35.9843687]]))*np.array([0.005,0.005*8, 0.005])
    #num_mesh = F.shape[0]
    rigid_body_3 = Rigidbody(num_mesh_1,2)
    
    rho_ = 1*np.ones(num_mesh_1)  # Assign uniform density
    
    rigid_body_3.Initialize_geometry(cor_3, rho_)
    print(rigid_body_3.position)

    v_3 = np.array([0.0, 0.0, 0.0])
    L_3 = np.array([0.0, -0.0, 10.0])
    #L_3 = np.array([0.0 , 8.0*np.sin(theta),  -8.0*np.cos(theta)])
    #L_3=np.zeros(3)
    R_ = np.eye(3)
    
    R_3 = np.array([
        [1.0, 0.0, 0.0],
        [0.0,np.sin(theta),np.cos(theta)],
        [0.0,-np.cos(theta),np.sin(theta)]
    ])
    R_3=np.eye(3)
    rigid_body_3.Initialize_dynamic(v_3, L_3, R_3,dt)
    
    
    my_multiple_body = multiple_body([rigid_body_1,rigid_body_2,rigid_body_3])
    #print(rigid_body_1.velocities)

    constraint_hinge_1=Hinge_Constraint(rigid_body_1,rigid_body_2,np.array([2.72835145e-02,0.0,9.37031485e-02])-rigid_body_1.position,np.array([2.72835145e-02,0.0,9.37031485e-02]),np.array([2.99535359e-02,0.0,-6.86618395e-02])-rigid_body_1.position,np.array([2.99535359e-02,0.0,-6.86618395e-02]))
    my_multiple_body.add_constraint(constraint_hinge_1)
    #print((np.array([[99.9960021972656250,9.4219951,22.2520008087158203]])-np.array([[94.00529501,9.42199516,35.9843687]]))*np.array([0.005,0.005*8, 0.005]))
    #constraint_hinge_2=Hinge_Constraint(rigid_body_2,rigid_body_3,np.array([-2.94742213e-02,0.0,1.01415666e-01]),np.array([-2.94742213e-02,0.0,1.01415666e-01])-rigid_body_3.position, np.array([-2.86112224e-02,0.0,-4.83662159e-02]),np.array([-2.86112224e-02,0.0,-4.83662159e-02])-rigid_body_3.position)
    constraint_hinge_2=Hinge_Constraint(rigid_body_2,rigid_body_3,np.array([-2.72835145e-02,0.0,9.37031485e-02]),np.array([-2.72835145e-02,0.0,9.37031485e-02])-rigid_body_3.position,np.array([-2.99535359e-02,0.0,-6.86618395e-02]),np.array([-2.99535359e-02,0.0,-6.86618395e-02])-rigid_body_3.position)
    my_multiple_body.add_constraint(constraint_hinge_2)
    #constraint_hinge_3=Hinge_Constraint(rigid_body_1,rigid_body_3,-rigid_body_1.position, -rigid_body_3.position,np.zeros(3),np.zeros(3))
    #my_multiple_body.add_constraint(constraint_hinge_3)

    steps = 1000

    my_multiple_body.simulate_with_polyscope(steps)


if __name__ == "__main__":
    main()

        






    

