#/bin/python3


# FOSS libraries
#Numerics
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as sclg

import multiprocessing as mp
import gc

# my stuff
import pyquad as pq
import P2FE, P1FE
from mesh import FlowMesh as FM
#from mesh import Flow_Mesh as FM


#Plotting
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# description of the reference triangle
ref_triangle = np.array([[0.,0.],[1.,0.],[0.,1.],
                        [0.5,0.0],[0.5,0.5],[0.0,0.5]])


def chunks(l, n):
    """Splits a list l into n almost equal chunks"""
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
    


class Diff_Operator(object):
    def __init__(self, mesh:FM, dt:float, alpha:float, 
                rho:float, mu:float, length:float, expansion:float, gravity:float):
        
        self.mesh = mesh
        self.dt = dt
        self.alpha = min(max(alpha, 0.0),1.0) 
        
        self.rho = rho
        self.mu = mu
        self.length = length
        self.expansion = expansion
        self.gravity = gravity

        self.stiff = scsp.csr_matrix((1,1))
        self.mass  = scsp.csr_matrix((1,1))
        self.conv  = scsp.csr_matrix((1,1))
        self.pdivv = scsp.csr_matrix((1,1))


        self.lgr = scsp.csr_matrix((1,1))
        self.lgr_rhs = np.zeros(1)
        self.forcing = np.zeros(1) # needs to be multiplied by buoyancy
        
        self.all_built = False
        
        self.operator = None
        self.iterative = False
        
        self.dirichlet_vals = {}  # keyed by boundary number
        self.subdomain_diff_factor = {} # keyed by element number
        
        self.forcing = None

    def check_element_areas(self):
        """Needed to check if there are backwards ordered elements in the mesh.
        Returns list of bad element numbers.
        """
        bad_els = []
        area = 0.
        mesh = self.mesh
        for el in range(mesh.els.shape[0]):
            if mesh.els[el,0]==6:
                el_area = sum(P2FE.local_forcing_el(mesh.get_element_node_coords(el),
                                                    mesh.gauss_points))
                if el_area<0.:
                    bad_els.append(el)
                    
                area += el_area
                
        print(len(bad_els), " bad elements")
                
        print("Total area", area)
        return bad_els
        
    def repair_bad_els(self, bad_els:list):
        """A list of element numbers with bad ordering can be used to reverse
        said ordering and make a mesh whose area is correct.
        """
        for el in bad_els:
            self.mesh.reverse_element_ordering(el)
        self.check_element_areas()
        
        
    def build_mass_para(self):
        """Parallel construction of mass matrix.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        nnodes = mesh.nnodesP2
        els = mesh.els
        
        # first build all local matrices
        worklist = []
        for el in range(els.shape[0]):
            if els[el,0]==6:  # only integrate if it is a second order triangle, not a point or line.
                worklist.append([el, 
                                mesh.gauss_points, 
                                mesh.get_element_node_coords(el)])
        
        with mp.Pool(mp.cpu_count()-1) as p:
            mass_locs = p.map(local_mass_mat, worklist) 

        # now allocate local entries to global entries

        ndl = self.mesh.node_dof_lookup_P2
        worklist2 = []
        for i in range(len(worklist)):
            mloc = mass_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs_P2(el)
            worklist2.append((mloc, (nnodes*2,nnodes*2), glob_dofs))
            
        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global_P2, worklist2)
            del worklist2
        gc.collect()
        
        return sum(results).tocsr()
    
    def build_stiff_para(self):
        """Parallel construction of stiffness matrix.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        nnodes = mesh.nnodesP2*2
        els = mesh.els
        
        # first build all local matrices
        worklist = []
        for el in range(els.shape[0]):
            if els[el,0]==6:  # only integrate if it is a second order triangle, not a point or line.
                worklist.append([el, mesh.gauss_points, 
                                mesh.get_element_node_coords(el)])
        
        with mp.Pool(mp.cpu_count()-1) as p:
            stiff_locs = p.map(local_stiff_mat, worklist) 
        # now allocate local entries to global entries
        ndl = self.mesh.node_dof_lookup_P2
        worklist2 = []
        for i in range(len(worklist)):
            loc = stiff_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs_P2(el)
            worklist2.append((loc, (nnodes,nnodes), glob_dofs))
            
        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global_P2, worklist2)
            del worklist2
        gc.collect()
        return sum(results).tocsr()

    def build_conv_para(self, ucoeffs:np.array):
        """WATCHOUT: ucoeffs should be the x and y components of velocity
        ordered by index i in node number of the MESH(!) 
        """
        
        mesh = self.mesh
        nodes = mesh.nodes
        nnodes = mesh.nnodesP2*2
        els = mesh.els
        
        # first build all local matrices
        worklist = []
        for el in range(els.shape[0]):
            if els[el,0]==6:  # only integrate if it is a second order triangle, not a point or line.
                ucsxy = np.array([ ucoeffs[n,:] for n in els[el,1:] ]) # extract relevant ucoeffs
                worklist.append([el, mesh.gauss_points, 
                                mesh.get_element_node_coords(el), ucsxy])
        
        with mp.Pool(mp.cpu_count()-1) as p:
            conv_locs = p.map(local_conv_mat, worklist) 
            
        # local to global
        ndl = self.mesh.node_dof_lookup_P2
        worklist2 = []
        for i in range(len(worklist)):
            loc = conv_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs_P2(el)
            worklist2.append((loc, (nnodes,nnodes), glob_dofs))
            
        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global_P2, worklist2)
            del worklist2
        gc.collect()
        return sum(results).tocsr()

    def build_pdivv_para(self):
        """Build the matrix that uses pressure to force incompressibility.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        nnodesP2 = mesh.nnodesP2
        nnodesP1 = mesh.nnodesP1
        els = mesh.els
        
        # first build all local matrices
        worklist = []
        for el in range(els.shape[0]):
            if els[el,0]==6:  # only integrate if it is a second order triangle, not a point or line.
                worklist.append([el, mesh.gauss_points, 
                                mesh.get_element_node_coords(el)])

        # list is built now process
        with mp.Pool(mp.cpu_count()-1) as p:
            pdivv_locs = p.map(local_pdivv_mat, worklist)
        
        
        worklist2 = []
        for i in range(len(worklist)):
            pdloc = pdivv_locs[i]
            el = worklist[i][0]
            glob_dofsP2 = mesh.get_element_global_dofs_P2(el)
            glob_dofsP1 = mesh.get_element_global_dofs_P1(el)

            
            worklist2.append([pdloc, (nnodesP1, nnodesP2*2), 
                                glob_dofsP2, glob_dofsP1])

        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-2) as p:
                results = p.map(mesh.local_to_global_P1, worklist2)
        
        return sum(results).tocsr()

    def boundary_lagrangian(self):
        """Uses a Lagrangian multiplier to set the BCs
        """
        mesh    = self.mesh
        tfn     = mesh.nnodes_dc
        row     = 0
        ndl     = mesh.node_dof_lookup_P2      
        
        lgr     = scsp.lil_matrix((tfn*2, mesh.nnodesP2*2))
        lgr_rhs = np.zeros(tfn*2)
        

        for node in mesh.dirich_nodes:
            dof_i = ndl[node,0]
            lgr[row, dof_i] = 1.
            lgr_rhs[row]    = self.dirichlet_vals[mesh.dirich_nodes[node]][0]
            row += 1
            
            dof_i = ndl[node,1]
            lgr[row, dof_i] = 1.
            lgr_rhs[row]    = self.dirichlet_vals[mesh.dirich_nodes[node]][1]
            row += 1
            
        self.lgr        = lgr
        self.lgr_rhs    = lgr_rhs

    def assemble_all(self, u_vec:np.array):
        """alpha is the variable defining implicit-explicitness. Use 1.0 for fully implicit.
        """
        if len(u_vec.shape)!=1:
            if u_vec.shape[1] != 1:
                raise IndexError("u_vec should be in dof_mapping format (single column)")
        mesh = self.mesh
        alpha   = self.alpha
        tfn = mesh.nnodes_dc

        rho = self.rho
        mu = self.mu
        L = self.length

        mass    = self.build_mass_para()
        stiff   = self.build_stiff_para()
        conv    = self.build_conv_para(mesh.map_solution_back(u_vec))
        pdivv   = self.build_pdivv_para()
        
        # save these
        self.mass = mass
        self.stiff = stiff
        self.conv = conv
        self.pdivv = pdivv

        self.boundary_lagrangian()
        lgr = self.lgr
        self.all_built = True
        dt = self.dt

        # kg.m.s/m².s² * m³/kg = m²/s 
        operator = mass+alpha*dt*(mu*stiff/rho*L**2/2.+conv) # P2 part of matrix

        p1_lgr = scsp.vstack([-pdivv/rho*dt,lgr])
        operator = scsp.vstack([operator,p1_lgr])

        zero_block = scsp.csr_matrix((p1_lgr.shape[0],p1_lgr.shape[0]))
        self.operator = scsp.hstack([operator, 
                                scsp.vstack([p1_lgr.transpose(), zero_block])]).tocsr()


    def update_conv(self, last_u:np.array):
        """Given the last solution vector, generates a new convection matrix. 
        
        This should be vectorised.
        """
        if len(last_u.shape)!=1:
            if last_u.shape[1] != 1:
                raise IndexError("u_vec should be in dof_mapping format (single column)")
        mesh = self.mesh
        alpha   = self.alpha

        rho = self.rho
        mu = self.mu
        L = self.length

        conv    = self.build_conv_para(mesh.map_solution_back(last_u))
        self.conv = conv
        
        mass = self.mass  
        stiff =self.stiff
        pdivv =self.pdivv 
        lgr = self.lgr
        
        dt = self.dt
        
        # kg.m.s/m².s² * m³/kg = m²/s 
        operator = mass+alpha*dt*(mu*stiff/rho*L**2/2.+conv) # P2 part of matrix

        p1_lgr = scsp.vstack([-pdivv/rho*dt,lgr])
        operator = scsp.vstack([operator,p1_lgr])

        zero_block = scsp.csr_matrix((p1_lgr.shape[0],p1_lgr.shape[0]))
        self.operator = scsp.hstack([operator, 
                                scsp.vstack([p1_lgr.transpose(), zero_block])]).tocsr()

    
    def calc_rhs_op_part(self, uold_vec:np.array):
        """Using alpha as the implicit-explicit factor, this method builds the 
        RHS operator)
        
        """
        if self.all_built:
            rho = self.rho
            mu = self.mu
            L = self.length
            alpha = self.alpha
            dt = self.dt
            RHSOP = self.mass - (1.-alpha)*dt*(mu*self.stiff/rho*L**2/2.+self.conv)
        else:
            raise TypeError("Assemble the matrices before doing this")
            
        return RHSOP.dot(uold_vec[:self.mesh.nnodesP2*2])

    def solve_one_step(self, vel:float, last_u_vec:np.array, forcing=None):
        """Bundle of stuff to solve one time step and return solution.
        """
        L = self.operator
        mesh = self.mesh
        if forcing is None:
            forcing = np.zeros(self.mesh.nnodesP2*2)

        rhs = np.zeros(mesh.nnodesP2*2+mesh.nnodesP1+mesh.nnodes_dc*2)
        rhs[:mesh.nnodesP2*2] = self.calc_rhs_op_part(last_u_vec)+forcing 
        rhs[mesh.nnodesP2*2+mesh.nnodesP1:] = self.lgr_rhs*vel

        if self.iterative:
                x, failed = sclg.gmres(L, rhs, restart=30)
                if not failed:
                    return x
                else: 
                    raise ValueError("did not converge")
        else :
            return sclg.spsolve(L, rhs)

    def calc_buoyancy(self, t_nodes:np.array):
        """t_nodes should be the values of the temperature at each dof
        mapping is by dof not by node number
        """
        if self.forcing is not None:
            buoyancy_forcing = -t_nodes*self.forcing*self.expansion*self.gravity
        else:
            self.calc_global_forcing()
            buoyancy_forcing = -t_nodes*self.forcing*self.expansion*self.gravity
        for node in self.mesh.dirich_nodes:
            dof_x, dof_y = self.mesh.node_dof_lookup_P2[node,:]
            buoyancy_forcing[dof_y] = 0.
            
        return buoyancy_forcing
                
    def calc_global_forcing(self):
        """Integration of basis functions for forcing purposes. Only need 
        to do this once, unless the nodes move.
        """
        mesh = self.mesh
        els = mesh.els
        forcing = np.zeros(shape=(mesh.nnodesP2*2))
        for el in range(els.shape[0]):
            if els[el,0]==6:
                elnodes = mesh.els[el,1:]
                loc = P2FE.local_forcing_el(mesh.get_element_node_coords(el), 
                                            mesh.gauss_points)
                for i in range(6):
                    node_j = elnodes[i]
                    if node_j not in mesh.dirich_nodes:
                        # only the y coord matters, otherwise we need 
                        # to use a gravity vector to know which way
                        # is gravity pointing.
                        dof_j = mesh.node_dof_lookup_P2[node_j, 1]
                        forcing[dof_j] += loc[i]
                
        self.forcing = forcing 
        
"""
---------------------------------------------------------------------------------
| Stiffness Matrix - Parallel Version                                           |
---------------------------------------------------------------------------------
"""          
def pointwise_stiff(xi_eta, node_coords):
    """This returns the block matrix above [[Axx, Ayx],[Axy,Ayy]] for a single point.
    the factor mu/rho is excluded
    """
    
    grads = P2FE.local_node_derivs(xi_eta, node_coords).transpose()
    K = np.zeros(shape=(12,12))
    # use numpy's outer product function
    # upper left
    K[:6,:6] = 4* np.outer(grads[0,:],grads[0,:]) + 2*np.outer(grads[1,:],grads[1,:])
    # lower right
    K[6:,6:] = 4* np.outer(grads[1,:],grads[1,:]) + 2*np.outer(grads[0,:],grads[0,:]) 
    #lower left
    K[:6,6:] = 2*np.outer(grads[0,:],grads[1,:])
    # upper right
    K[6:,:6] = 2*np.outer(grads[1,:],grads[0,:])
    
    return K*np.linalg.det(P2FE.jacobian(xi_eta, node_coords))
    
def local_stiff_mat(args):
    elnum, int_points, el_node_coords = args
    """Integrates over the element given the integration points and
    the element coords to get the local stiffness. 
    Does not include factor mu/2"""
    npoints = int_points.shape[0]
    
    K = np.zeros(shape=(12,12))
    for i in range(npoints):
        K += pointwise_stiff(int_points[i,0:2],el_node_coords)*int_points[i,-1]
        
    return K


"""
---------------------------------------------------------------------------------
| Mass Matrix   - Parallel Version                                              |
---------------------------------------------------------------------------------
"""
def local_mass_matrix_P2_point(xi_eta:np.array, el_node_coords:np.array):
    sh_fkt = P2FE.shape_func_vec(xi_eta)
    C_dyad = np.outer(sh_fkt,sh_fkt)
    M=np.zeros(shape=(12,12))
    
    M[:6,:6] = C_dyad
    M[6:,6:] = C_dyad
    return M*np.linalg.det(P2FE.jacobian(xi_eta, el_node_coords))
    
def local_mass_mat(args):
    eln, int_points, el_node_coords = args
    M = np.zeros(shape=(12,12))
    for i in range(int_points.shape[0]):
        M += local_mass_matrix_P2_point(int_points[i,0:2], 
                                        el_node_coords)*int_points[i,-1]
    return M

"""
---------------------------------------------------------------------------------
| Convection Matrix    Parallel Version                                         |
---------------------------------------------------------------------------------
"""

def local_conv_matrix_P2_point(xi_eta:np.array, el_node_coords:np.array, ucoeffs):
    sh_fkt = P2FE.shape_func_vec(xi_eta)
    grads = P2FE.local_node_derivs(xi_eta, el_node_coords) 
    u_here_x = sh_fkt.dot(ucoeffs[:,0])
    u_here_y = sh_fkt.dot(ucoeffs[:,1])
    
    C=np.zeros(shape=(12,12))
    Cd = np.outer(sh_fkt, u_here_x*grads[:,0]+u_here_y*grads[:,1] )
    C[:6,:6] = C[6:,6:] = Cd

    return C*np.linalg.det(P2FE.jacobian(xi_eta, el_node_coords))



def local_conv_mat(args):
    el, int_points, el_node_coords, ucoeffs = args
    C = np.zeros(shape=(12,12))
    for i in range(int_points.shape[0]):
        C += local_conv_matrix_P2_point(int_points[i,0:2], 
                                        el_node_coords, ucoeffs)*int_points[i,-1]
    return C


"""
---------------------------------------------------------------------------------
| Pressure  Matrix                                                               |
---------------------------------------------------------------------------------
"""
def Pdivv_mat_point(xi_eta:np.array, el_node_coords:np.array):
    grads = P2FE.local_node_derivs(xi_eta, el_node_coords) 
    v_vec = np.hstack((grads[:,0],grads[:,1]))
    p_vec = P1FE.shape_func_vec_P1(xi_eta)
    
    mat = np.outer(p_vec,v_vec)*np.linalg.det(P2FE.jacobian(xi_eta, el_node_coords))
    return mat
    

def local_pdivv_mat(args):
    el, int_points, el_node_coords= args
    npoints = int_points.shape[0]
    
    Pdivv_mat = np.zeros(shape=(3,12))
    for i in range(int_points.shape[0]):
        Pdivv_mat += Pdivv_mat_point(int_points[i,0:2], el_node_coords)*int_points[i,-1]
    return Pdivv_mat
    


"""
---------------------------------------------------------------------------------
| Graphing stuff                                                                |
---------------------------------------------------------------------------------
"""



def graph_solution(ucoeffs, mesh:FM, t:float, 
                         scale=20.):
    """Generates a plot of the solution.
    """
    nodes = mesh.nodes
    els = mesh.els
    
    u_x = ucoeffs[:,0]
    u_y = ucoeffs[:,1]
    scaling = ((mesh.nodes[:,1].max()-mesh.nodes[:,1].min())
                      /(mesh.nodes[:,0].max()-mesh.nodes[:,0].min()))#**0.25
    vol_els = [a[1:4] for a  in mesh.els if a[0]==6 ]
    tris = mtri.Triangulation(nodes[:,0],nodes[:,1],np.vstack(vol_els))
    
    dpi = 75
    width = 3200/dpi
    height = scaling*3200*1.33/dpi
    height += height%2
    fig1 = plt.figure(figsize=(width,height))
    ax1 = fig1.add_subplot(111)  
    
    v = np.linspace(0., scale   ,100)
    speed = np.sqrt((u_x**2+u_y**2))
    ysss = ax1.tricontourf(tris, speed, v, cmap='viridis')
    
    
    ax1.set_aspect('equal')
    
    ax1.triplot(tris, color=(0.5,0.5,0.5,0.5))
    
    if t is None:
        ax1.set_title('Temperature')
    else:
        ax1.set_title('Temperature at t='+f"{t:2.03f} seconds")
    
    # Plots direction of the electrical vector field
    for i in range(speed.shape[0]):
        if speed[i] == 0. :
            speed[i] = 1.
    
    ax1.quiver(tris.x, tris.y, u_x/speed, u_y/speed,
              units='xy', scale=50., zorder=3, color=(0.75,0.75,0.75,0.75),
              width=0.007, headwidth=1., headlength=3.)

    cbar = fig1.colorbar(ysss, orientation='horizontal')
    fig1.tight_layout()
    
    return fig1
 
