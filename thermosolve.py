#/bin/python3


# FOSS libraries
#Numerics
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as sclg

import multiprocessing as mp

#Plotting
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#Python-OS interaction
import os
import pickle as pkl
import gc

# my stuff
import pyquad as pq
import P2FE
from mesh import ThermoMesh as TM
#from mesh import Flow_Mesh as FM


# description of the reference triangle
ref_triangle = np.array([[0.,0.],[1.,0.],[0.,1.],
                        [0.5,0.0],[0.5,0.5],[0.0,0.5]])


def chunks(l, n):
    """Splits a list l into n almost equal chunks"""
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
    

class Diff_Operator(object):
    def __init__(self, mesh:TM, dt:float, alpha:float):
        self.mesh = mesh
        self.dt = dt
        self.alpha = min(max(alpha, 0.0),1.0) 
        
        
        
        self.stiff = scsp.csr_matrix((1,1))
        self.mass = scsp.csr_matrix((1,1))
        self.conv = scsp.csr_matrix((1,1))
        
        self.lgr = scsp.csr_matrix((1,1))
        self.lgr_rhs = np.zeros(1)
        self.operator = None
        
        self.all_built = False
        
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
                
        print("Total area", area)
        return bad_els
        
    def repair_bad_els(self, bad_els:list):
        """A list of element numbers with bad ordering can be used to reverse
        said ordering and make a mesh whose area is correct.
        """
        for el in bad_els:
            self.mesh.reverse_element_ordering(el)
        self.check_element_areas()
        
    def allocate_element_diffusion_factors(self, factor_dict:dict):
    
        # W/m.k  / (J/Kg.k) / kg/m3  
        # J/(s.m.K) * (Kg.K.m3)/(Jkg) = m2/s
        #factor =  self.kappa /self.heat_cap / self.rho * self.length**2
        for el in self.mesh.subdomains:
            subdom  = self.mesh.subdomains[el]
            self.subdomain_diff_factor[el] = factor_dict[subdom]
                
        
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
        ndl = self.mesh.node_dof_lookup
        worklist2 = []
        for i in range(len(worklist)):
            mloc = mass_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs(el)
            worklist2.append((mloc, (nnodes,nnodes), glob_dofs))
            
        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global, worklist2)
            del worklist2
        gc.collect()
        
        return sum(results).tocsr()
    
    def build_stiff_para(self):
        """Parallel construction of stiffness matrix.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        nnodes = mesh.nnodesP2
        els = mesh.els
        
        # first build all local matrices
        worklist = []
        for el in range(els.shape[0]):
            if els[el,0]==6:  # only integrate if it is a second order triangle, not a point or line.
                factor = self.subdomain_diff_factor[el]                
                worklist.append([el, mesh.gauss_points, 
                                mesh.get_element_node_coords(el), factor])
        
        with mp.Pool(mp.cpu_count()-1) as p:
            stiff_locs = p.map(local_stiff_mat, worklist) 
        print("thermal stiffness worklist done")
        # now allocate local entries to global entries
        ndl = self.mesh.node_dof_lookup
        worklist2 = []
        for i in range(len(worklist)):
            loc = stiff_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs(el)
            worklist2.append((loc, (nnodes,nnodes), glob_dofs))
            
        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global, worklist2)
            del worklist2
        gc.collect()
        return sum(results).tocsr()


    def build_conv_para(self, ucoeffs:np.array):
        """WATCHOUT: ucoeffs should be the x and y components of velocity
        ordered by index i in node number of the THERMAL MESH(!) 
        """
        
        mesh = self.mesh
        nodes = mesh.nodes
        nnodes = mesh.nnodesP2
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
        ndl = self.mesh.node_dof_lookup
        worklist2 = []
        for i in range(len(worklist)):
            loc = conv_locs[i]
            el = worklist[i][0]    
            glob_dofs = mesh.get_element_global_dofs(el)
            worklist2.append((loc, (nnodes,nnodes), glob_dofs))

        del worklist
        worklist2 = chunks(worklist2, 2*(mp.cpu_count()-1))
        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(mesh.local_to_global, worklist2)
            del worklist2
        gc.collect()
        return sum(results).tocsr()

    def assemble_all(self, ucoeffs:np.array):
        """alpha is the variable defining implicit-explicitness. Use 1.0 for fully
         implicit. ucoeffs should be nx2
        """
        alpha = self.alpha
        mass = self.build_mass_para()
        stiff = self.build_stiff_para()
        conv = self.build_conv_para(ucoeffs)
        
        # save these
        self.mass = mass
        self.stiff = stiff
        self.conv = conv
        self.boundary_lagrangian()
        self.all_built = True
        dt = self.dt
        
        
        operator = mass + alpha*dt*(conv + stiff)
        operator = scsp.vstack([operator, self.lgr])
        lgr_vert = scsp.vstack([self.lgr.transpose(), 
                                scsp.csr_matrix((self.lgr.shape[0],self.lgr.shape[0]))])
        self.operator = scsp.hstack([operator, lgr_vert]).tocsr()
        
        
    def assemble_rhs_op(self, told:np.array):
        """Using alpha as the implicit-explicit factor, this method builds the 
        RHS operator)
        
        """
        if self.all_built:
            alpha = self.alpha
            dt = self.dt
            RHSOP = self.mass - (1.-alpha)*dt*(self.conv + self.stiff)
        else:
            raise TypeError("Assemble the matrices before doing this")
            
        return RHSOP.dot(told[:self.mesh.nnodesP2])
        
    def boundary_lagrangian(self):
        """Uses a Lagrangian multiplier to set the BCs
        """
        mesh    = self.mesh
        tfn     = mesh.nnodes_dc
        row     = 0
        dl      = mesh.node_dof_lookup        
        
        lgr     = scsp.lil_matrix((tfn, mesh.nnodesP2), dtype=np.float)
        lgr_rhs = np.zeros(tfn)
        
        for node in mesh.dirich_nodes:
            dof_i = dl[node]
            lgr[row, dof_i] = 1.
            lgr_rhs[row]    = self.dirichlet_vals[mesh.dirich_nodes[node]]
            row += 1
            
        self.lgr        = lgr
        self.lgr_rhs    = lgr_rhs
        
    def solve_one_step(self, temp_diff:float, told:np.array):
        """Bundle of stuff to solve one time step and return solution.
        """
        L = self.operator
        rhs = np.hstack([self.assemble_rhs_op(told)+self.forcing, 
                        self.lgr_rhs*temp_diff])

        if self.iterative:
                x, failed = sclg.gmres(L, rhs, restart=30)
                if not failed:
                    return x
                else: 
                    raise ValueError("did not converge")
        else :
            return sclg.spsolve(L, rhs)
            
  
            
    def initial_condition(self, baseval:float, tdiff:float):
        """Multiply by temp diff after.
        """
        
        mesh = self.mesh
        init_cond = np.zeros(mesh.nnodesP2) + baseval
        dl = mesh.node_dof_lookup  
            
        for node in mesh.dirich_nodes:
            init_cond[dl[node]] = self.dirichlet_vals[mesh.dirich_nodes[node]]*tdiff
            
        return init_cond

    def update_convection(self, ucoeffs:np.array):
        """alpha is the variable defining implicit-explicitness. Use 1.0 for 
        fully implicit.
        """
        alpha = self.alpha
        self.conv = self.build_conv_para(ucoeffs)
        
        # save these
        mass = self.mass 
        stiff = self.stiff 
        conv = self.conv
        self.all_built = True
        dt = self.dt
        # W/m.k  / (J/Kg.k) / kg/m3  
        # J/(s.m.K) * (Kg.K.m3)/(Jkg) = m2/s
        
        
        operator = mass + alpha*dt*(conv + stiff)
        operator = scsp.vstack([operator, self.lgr])
        lgr_vert = scsp.vstack([self.lgr.transpose(), 
                                scsp.csr_matrix((self.lgr.shape[0],self.lgr.shape[0]))])
        self.operator = scsp.hstack([operator, lgr_vert]).tocsr()
        
    def reassemble(self):
        
        if self.all_built:
            mass = self.mass 
            stiff = self.stiff 
            conv = self.conv
            dt = self.dt
            alpha = self.alpha
            
            operator = mass + alpha*dt*(conv + stiff)
            operator = scsp.vstack([operator, self.lgr])
            lgr_vert = scsp.vstack([self.lgr.transpose(), 
                                    scsp.csr_matrix((self.lgr.shape[0],self.lgr.shape[0]))])
            self.operator = scsp.hstack([operator, lgr_vert]).tocsr()
        else:
            raise ValueError("can't reassemble before building")
        
    def calc_global_forcing(self, elnums):
        """Integration of basis functions for forcing purposes. Only need 
        to do this once, unless the nodes move.
        """
        mesh = self.mesh
        els = mesh.els
        forcing = np.zeros(shape=(mesh.nnodesP2))
        for el in range(els.shape[0]):
            if els[el,0]==6:
                elnodes = mesh.els[el,1:]
                loc = P2FE.local_forcing_el(mesh.get_element_node_coords(el), 
                                            mesh.gauss_points)
                for i in range(6):
                    
                    node_i = elnodes[i]
                    dof_i = mesh.node_dof_lookup[node_i]
                    forcing[dof_i] += loc[i]
                
        self.forcing = forcing 
        
    
    def calc_area(self):
        area = 0.
        mesh = self.mesh
        for el in range(mesh.els.shape[0]):
            if mesh.els[el,0]==6:
                area += sum(P2FE.local_forcing_el(mesh.get_element_node_coords(el), 
                                                mesh.gauss_points))
        return area
        
        
        
"""
---------------------------------------------------------------------------------
| Mass Matrix   - Parallel Version                                              |
---------------------------------------------------------------------------------
"""
def local_mass_point(xi_eta:np.array, el_node_coords:np.array):
    """Local Mass matrix for a single gaussian point. 
    """
    sh_fkt = P2FE.shape_func_vec(xi_eta)
    M = np.outer(sh_fkt,sh_fkt)

    return M*np.linalg.det(P2FE.jacobian(xi_eta, el_node_coords))
    

def local_mass_mat(args):
    """Local mass matrix for given element
    """
    eln, int_points, el_node_coords = args
    M = np.zeros(shape=(6,6))
    for i in range(int_points.shape[0]):
        M += local_mass_point(int_points[i,0:2], 
                              el_node_coords)*int_points[i,-1]
    return M
    
"""
---------------------------------------------------------------------------------
| Stiffness Matrix - Parallel Version                                           |
---------------------------------------------------------------------------------
"""          
    
def local_stiff_point(xi_eta, node_coords):
    """Local stiffness matrix for a single integration point.
    """
    grads = P2FE.local_node_derivs(xi_eta, node_coords).transpose()
    # use numpy's outer product function
    K= np.outer(grads[0,:],grads[0,:]) + np.outer(grads[1,:],grads[1,:])
    
    return K*np.linalg.det(P2FE.jacobian(xi_eta, node_coords))
    
def local_stiff_mat(args):
    """Integrates over the element given the integration points and
    the element coords to get the local stiffness. 
    """
    elnum, int_points, el_node_coords, factor = args
    npoints = int_points.shape[0]
    
    K = np.zeros(shape=(6,6))
    for i in range(npoints):
        K += local_stiff_point(int_points[i,0:2],el_node_coords)*int_points[i,-1]
        
    return K*factor


"""
---------------------------------------------------------------------------------
| Convection Matrix    Parallel Version                                         |
---------------------------------------------------------------------------------
"""
def local_conv_point(xi_eta, node_coords, ucoeffs):
    """Convection matrix values for a single integration point
    """
    
    sh_fkt = P2FE.shape_func_vec(xi_eta)
    grads = P2FE.local_node_derivs(xi_eta, node_coords) 
    u_here_x = sh_fkt.dot(ucoeffs[:,0])
    u_here_y = sh_fkt.dot(ucoeffs[:,1])

    Cd = np.outer(sh_fkt, u_here_x*grads[:,0]+u_here_y*grads[:,1] )#.transpose()
    return Cd*np.linalg.det(P2FE.jacobian(xi_eta, node_coords))
    

def local_conv_mat(args):
    """Convection matrix for a given element.
    """
    el, int_points, el_node_coords, ucoeffs = args
    C = np.zeros(shape=(6,6))
    for i in range(int_points.shape[0]):
        C += local_conv_point(int_points[i,0:2], 
                              el_node_coords, ucoeffs)*int_points[i,-1]
    return C
    
"""
---------------------------------------------------------------------------------
| Graphing stuff                                                                |
---------------------------------------------------------------------------------
"""

def graph_solution(ucoeffs,  tsol, mesh:TM, t:float, 
                         scale=20.):
    """Produces a plot of the solution.
    """
    nodes = mesh.nodes
    els = mesh.els

    tsol = mesh.map_solution_back(tsol)
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
    if scale is float:
        v = np.linspace(-scale, scale   ,100)
    else: # give scale as (min,max)
        v = np.linspace(scale[0], scale[1]   ,100)
    ysss = ax1.tricontourf(tris, tsol, v, cmap='bwr')
    
    
    ax1.set_aspect('equal')
    
    ax1.triplot(tris, color=(0.5,0.5,0.5,0.5))
    
    if t is None:
        ax1.set_title('Temperature')
    else:
        ax1.set_title('Temperature at t='+f"{t:2.03f} seconds")
    
    # Plots direction of the electrical vector field
    sol_norm = np.sqrt((u_x**2+u_y**2))
    for i in range(sol_norm.shape[0]):
        if sol_norm[i] == 0. :
            sol_norm[i] = 1.
    
    ax1.quiver(tris.x, tris.y, u_x/sol_norm, u_y/sol_norm,
              units='xy', scale=50., zorder=3, color=(0.75,0.75,0.75,0.75),
              width=0.007, headwidth=1., headlength=3.)

    cbar = fig1.colorbar(ysss, orientation='horizontal')
    fig1.tight_layout()
    
    return fig1
