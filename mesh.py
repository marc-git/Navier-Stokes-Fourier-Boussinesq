#/bin/python3
"""This module contains two important classes:
 -1. Thermomesh for a P2 mesh in the thermal problem
 -2. FlowMesh for a Taylor-Hood mesh in the Flow problem
 
 both classes read from a second order triangle mesh generated in VTK format
 by GMSH. 
 
 The same VTK (ASCII) file can be used for both meshes. Area groups will be 
 allocated as subdomains, while edge groups will be allocated as nodes.
 
"""

# FOSS libraries
#Numerics
import numpy as np
import pyquad as pq
import scipy.sparse as scsp



#Plotting
import matplotlib.pyplot as plt
import matplotlib.tri as mtri



class ThermoMesh(object):
    """This class holds the mesh data.  It has methods to read the mesh file 
    and allocate DOFs accordingly.
    
    The mesh file should be built in GMSH in 2D with 2nd order triangular 
    elements.  Physical Groups should be made. 
    
    The mesh file should be exported as a VTK file in ASCII format.
    
     --- INITIALISING ---
    Initialise the object with a study name, and the path to the mesh file.
    
     --- LOADING ---
    load_mesh() will read the meshfile and interpret data. 
    graph_domain() to see if it looks right. Homogeneous Dirichlet Boundary 
        Nodes are in blue.
    node_dof_mapping() will allocate each node to a DOF number
    
     --- 
    """
    def __init__(self, vtkfile:str):
        self.vtkfile = vtkfile
        
        self.nnodesP2   = 0
        self.nels       = 0
        self.nnodes_dc  = 0
        self.nnodes_neu = 0
        
        self.nodes    = None  # will be numpy array
        self.els      = None  # will be numpy array
        
        self.subdomains = {}  # dictionary lookup by element number
        self.dirich_nodes = {}
        
        self.node_nums = [] # all relevant node numbers
        
        self.node_dof_lookup = None  # will be numpy array
        self.total_fixed_nodes = 0
        
        self.physical = {}
        
        self.gauss_points = None
        
    def load_mesh(self):
        """Parses the mesh in VTK legacy format (ASCII) as output by GMSH. 
        The mesh should be built specifically for thermal work.  Nodes are 
        presumed to match perfectly with the flow mesh.
        
        """
        with open(self.vtkfile, 'r') as f:
            print(f.readline())
            print(f.readline())
            print(f.readline())
            print(f.readline())
            a = str(f.readline())
            # read number of nodes from the line
            nnodes = int(a.split(' ')[1])
            self.nnodesP2 = nnodes
            # read all the node coords
            nodes = np.zeros(shape=(nnodes,2))
            for i in range(nnodes):
                x,y,_ = f.readline().split(' ')  # ignore z
                nodes[i,0] = float(x)
                nodes[i,1] = float(y)
            self.nodes = nodes.copy()

            print(f.readline())
            _, nels, p = f.readline().split(' ')
            print("Reading ", nels, " elements")

            els = np.zeros(shape=(int(nels),6+1),dtype=int)
            P2_nodes = []
            counters = {}
            for i in range(int(nels)):
                
                line = f.readline().split(' ')
                line = [int(a) for a in line]
                if line[0] in counters:
                    counters[line[0]] +=1
                else:
                    counters[line[0]] = 1
                for j in range(len(line)):
                    els[i,j] = line[j]
                if els[i,0] ==6: # add not yet added nodes to this list
                    for node in els[i,1:]:
                        if node not in P2_nodes:
                            if node==6:
                                print("added now from element", i)
                            P2_nodes.append(node)
                            
                        
            print("\n Element totals")
            for key in counters:
                print(key,"-node elements :", counters[key])
            self.node_nums = P2_nodes
            self.node_nums.sort()
            self.els = els.copy()
            print("Read ", nels,"elements")
            

            print("")
            print(1, f.readline())
            line = f.readline()
            print(line.split(' '))
            # skip the cell types
            blah, skip = line.split(' ')
            skip = int(skip)
            for i in range(skip):
                f.readline()
            
            # find out how many dirichlet elements
            print(1, f.readline())
            blah, num = f.readline().split(' ')
            print(f.readline())

            print(f.readline())
            
            
            boundaries = {}
            for i in range(int(num)):
                number = int(f.readline())
                if number != -1:
                    if els[i,0] == 6:
                        # then this is a subdomain description
                        self.subdomains[i] = number
                    else:
                        # this is a boundary condition tag
                        if number in boundaries.keys():
                            boundaries[number].append(i)
                        else:
                            boundaries[number] = []
                            boundaries[number].append(i)
                            
                        
            print(f.readline())
            print(f.readline())

        # collect BV nodes
        boundaries_by_node = {}
        for b_num in sorted(list(boundaries.keys())):
            for el in boundaries[b_num]:
                el_nodes = els[el,:]
                n = els[el,0]
                for i in range(1,1+n):
                    node = els[el,i]
                    # only add a node once
                    if node not in boundaries_by_node.keys():
                        boundaries_by_node[node] = b_num
                
        self.dirich_nodes = boundaries_by_node
        #count
        self.nnodesP2 = len(self.node_nums)
        self.nels     = int(nels)
        self.nnodes_dc= len(self.dirich_nodes.keys())
        print("Success!")
        print("Mesh has total ", self.nnodesP2, " nodes.")
        print(self.nnodes_dc, " have dirichlet BCs.")
        
        
    def graph_domain(self):
        """Returns a simple matplotlib graph of the domain triangulation and nodes"""
        
        vol_els = [a[1:4] for a  in self.els if a[0]==6 ]
        fig1 = plt.figure(figsize=(16,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_aspect('equal')
        
        tris = mtri.Triangulation(self.nodes[:,0],self.nodes[:,1],np.vstack(vol_els))
        ax1.triplot(tris, 'ko-', lw=1, ms=1.5)
        for i in range(1,10):
            dirichsx = [self.nodes[a,0] for a in self.dirich_nodes if self.dirich_nodes[a] == i]
            dirichsy = [self.nodes[a,1] for a in self.dirich_nodes if self.dirich_nodes[a] == i]
            if len(dirichsx) :
                ax1.plot(dirichsx,dirichsy, 'o',  label="Group"+str(i), ms=2.5)
            


        
        ax1.set_title('Domain')
        plt.tight_layout()
        plt.legend()
        return fig1

    def node_dof_mapping(self):
        """Maps nodes in simple order. This could be improved by implementing e.g. Cuthill-McKee 
        """
         
        count = 0
        self.node_dof_lookup = -np.ones(self.nodes.shape[0], dtype=int)
        
        for node in self.node_nums:
    
            self.node_dof_lookup[node] = count
            count += 1

        print(count, (self.nnodesP2))
        
        if count!= (self.nnodesP2):
            raise ValueError("number of nodes allocated does not match number of P2 nodes")
            
    def get_element_node_coords(self, elnum:int):
        """Returns the coordinates of the element nodes as a numpy array shaped
        (nnodes, ndim)
        """
        if elnum < 0:
            raise ValueError("negative element numbers?")
            
        if elnum >= self.nels:
            raise ValueError("Element index out of range")
        
        elnodes = self.els[elnum,:]
        node_coords = np.zeros(shape=(elnodes[0], 2))
        for i in range(1,elnodes[0]+1):
            node_coords[i-1,:] = self.nodes[elnodes[i],:]
        
        return node_coords    
    
    def get_element_global_dofs(self, elnum:int):
        """Returns the global DOFs belonging to a given element.
        """
        if elnum < 0:
            raise ValueError("negative element numbers?")
            
        if elnum >= self.nels:
            raise ValueError("Element index out of range")
            
        elnodes = self.els[elnum,:]
        glob_dofs = np.zeros(6, dtype=int)   
        for nod in range(elnodes[0]):
            glob_dofs[nod] = self.node_dof_lookup[elnodes[1:][nod]]
        return glob_dofs        
        
    
    def determine_gauss_points(self, polydeg:int, ndim=2):
        """Determine the gauss points on the reference triangle for a legendre 
        polynomial of degree polydeg, dimensions are set to two for these meshes.
        
        returns the gauss points in rows with last column being the associated weight.
        """
        
        gauss_points_and_weights = pq.gauss_map_to_simplex(pq.gauss_points_box(polydeg, ndim))
        
        vol = gauss_points_and_weights[:,-1].sum()
        if abs(vol-0.5) > 1e6: # error check
            raise ValueError("Problem with gauss weights. They don't add up"+
                            " to the volume 0.5")
            
        self.gauss_points = gauss_points_and_weights.copy()
        

    def local_to_global(self, argslist):
        """
        Local matrix is mapped with first all x than all y dofs. 
        """
        Kshape = argslist[0][1]
        
        K = scsp.lil_matrix(Kshape)
        
        for k in range(len(argslist)):
            local, ks, loc_nodes = argslist[k]
            for i in range(6):
                for j in range(6):
                    # could make this more efficient by recognising symmetry 
                    globi = loc_nodes[i]
                    globj = loc_nodes[j]
                    K[globi,globj] += local[i,j]
        return K

    def check_node_equivalence(self, nodes:np.array):
        """check if the mesh.nodes from another mesh object
            have all the same coordinates as this mesh.
        """
        if nodes.shape[1] < 2:
            raise ValueError("Wrong shape for nodes")
        if nodes.shape[0] < self.nodes.shape[0]:
            raise ValueError("Can't check equivalence because insufficient nodes in list")
        
        return (self.nodes == nodes[:self.nodes.shape[0],:]).min()
        

        
    def map_solution_back(self, sol_vec:np.array):
        """Returns the flow field as a Nx2 vector given input of the solution as a
        N-vector
        """
        sol = np.zeros(shape=(self.nodes.shape[0]))
        for node in self.node_nums:
            t = self.node_dof_lookup[node]
            sol[node] = sol_vec[t]
            
        return sol
        
    def reverse_element_ordering(self, elnum):
        """Sometimes GMSH will produce an element with 
        backwards ordering. This will reverse the order.
        """
        old = self.els[elnum,:].copy()
        new = np.array([old[0],
                        old[1], old[3], old[2], 
                        old[6], old[5], old[4]])
        
        self.els[elnum,:] = new.copy()
            
    def add_surface_to_dirichlet(self, subdomain:int, P1=True, overwrite=False):
        """It is necessary to set some whole elements to zero flow sometimes.
        This method adds a given subdomain to the dirichlet set.
        """
        new_dc_nodes = []
        # get the elements of interest
        els = [a for a in self.subdomains if self.subdomains[a]==subdomain]
        if P1 :
            k = 4
        else:
            k = 7
        for el in els:
            if self.els[el,0] ==6 :
                nodes = self.els[el,1:k]
                for i in range(k-1):
                    if nodes[i] not in new_dc_nodes:
                        new_dc_nodes.append(nodes[i])
                        
                        
            else:
                raise ValueError("Volume element "+str(el)+" is not a volume element")
        
        
        for node in new_dc_nodes:
            if node not in self.dirich_nodes or overwrite:
                self.dirich_nodes[node] = subdomain
                
        self.nnodes_dc = len(self.dirich_nodes)
        
    def remove_dirichlet_group(self, group_tuple:tuple):
        """Will eliminate an edge from the dirichlet group, effectively converting
        to a Neumann boundary unless something else is done.
        """
        to_remove = []
        for node in self.dirich_nodes:
            if self.dirich_nodes[node] in group_tuple:
                to_remove.append(node)
        print("Removing " + str(len(to_remove)) + " nodes")
        for node in to_remove:
            self.dirich_nodes.pop(node)
        self.nnodes_dc = len(self.dirich_nodes)
        
                
class FlowMesh(object):
    """rewrite of mesh class to include only mesh related things
    
    This class holds the mesh data.  It has methods to read the mesh file 
    and allocate DOFs accordingly.
    
    The mesh file should be built in GMSH in 2D with 2nd order triangular 
    elements.  Physical Groups should be made on all area and edge elements.
    
    The mesh file should be exported as a VTK file in ASCII format.
    
     --- INITIALISING ---
    Initialise the object with a study name, and the path to the mesh file.
    
     --- LOADING ---
    load_mesh() will read the meshfile and interpret data. 
    graph_domain() to see if it looks right. Homogeneous Dirichlet Boundary 
        Nodes are in blue.
    node_dof_mapping() will allocate each node to a DOF number
    
     --- 
    """
    def __init__(self, vtkfile:str):
        self.vtkfile = vtkfile
        
        self.nnodesP2 = 0
        self.nnodesP1 = 0
        self.nels     = 0
        self.nnodes_dc= 0
        
        self.nodes    = None
        self.els      = None
        
        self.subdomains = {}  # dictionary lookup by element number
        self.dirich_nodes = {}
        
        self.P1_node_nums = []
        self.P2_node_nums = []
        
        self.node_dof_lookup_P2 = None
        self.node_dof_lookup_P1 = None
        self.nnodes_dc = 0
        
        self.physical = {}
        
        self.gauss_points = None
        

    def load_mesh(self):
        """Parses the mesh in VTK legacy format (ASCII) as output by GMSH. 
        """
        with open(self.vtkfile, 'r') as f:
            print(f.readline())
            print(f.readline())
            print(f.readline())
            print(f.readline())
            a = str(f.readline())
            # read number of nodes from the line
            nnodes = int(a.split(' ')[1])
            # read all the node coords
            nodes = np.zeros(shape=(nnodes,2))
            for i in range(nnodes):
                x,y,_ = f.readline().split(' ')  # ignore z
                nodes[i,0] = float(x)
                nodes[i,1] = float(y)
            self.nodes = nodes.copy()

            print(f.readline())
            _, nels, p = f.readline().split(' ')
            print("Reading ", nels, " elements")

            els = np.zeros(shape=(int(nels),6+1),dtype=int)
            P1_nodes = []
            P2_nodes = []
            counters = {}
            for i in range(int(nels)):
                line = f.readline().split(' ')
                line = [int(a) for a in line]
                if line[0] in counters:
                    counters[line[0]] +=1
                else:
                    counters[line[0]] = 1
                for j in range(len(line)):
                    els[i,j] = line[j]
                if els[i,0] ==6: # add not yet added nodes to this list
                    for node in els[i,1:4]:
                        if node not in P1_nodes:
                            P1_nodes.append(node)
                        if node not in P2_nodes:
                            P2_nodes.append(node)
                    for node in els[i,4:]:
                        if node not in P2_nodes:
                            P2_nodes.append(node)
                        
            print("\n Element totals")
            for key in counters:
                print(key,"-node elements :", counters[key])
            
            self.P1_node_nums = P1_nodes
            self.P1_node_nums.sort()
            self.P2_node_nums = P2_nodes
            self.P2_node_nums.sort()
            self.els = els.copy()
            print("Read ", nels,"elements")
            
            print("")
            print(1, f.readline())
            line = f.readline()
            print(line.split(' '))
            # skip the cell types
            blah, skip = line.split(' ')
            skip = int(skip)
            for i in range(skip):
                f.readline()
            
            # find out how many dirichlet elements
            print(1, f.readline())
            blah, num = f.readline().split(' ')
            print(f.readline())

            print(f.readline())
            
            
            boundaries = {}
            for i in range(int(num)):
                number = int(f.readline())
                if number != -1:
                    if els[i,0] == 6:
                        # then this is a subdomain description
                        self.subdomains[i] = number
                    else:
                        # this is a boundary condition tag
                        if number in boundaries.keys():
                            boundaries[number].append(i)
                        else:
                            boundaries[number] = []
                            boundaries[number].append(i)
                            
                    
            print(f.readline())
            print(f.readline())

        # collect BV nodes
        
        # collect BV nodes
        boundaries_by_node = {}
        for b_num in sorted(list(boundaries.keys())):
            for el in boundaries[b_num]:
                el_nodes = els[el,:]
                n = els[el,0]
                for i in range(1,1+n):
                    node = els[el,i]
                    # only add a node once
                    if node not in boundaries_by_node.keys():
                        boundaries_by_node[node] = b_num
                
        self.dirich_nodes = boundaries_by_node
        
        #count
        self.nnodesP2 = len(self.P2_node_nums)
        self.nnodesP1 = len(self.P1_node_nums)
        self.nels     = int(nels)
        self.nnodes_dc= len(self.dirich_nodes.keys())
        print("Success!")
        print("Mesh has total ", self.nnodesP2, " nodes.")
        print(self.nnodesP1, " additional nodes are pressure nodes")
        print(self.nnodes_dc, " have homogeneous dirichlet BCs.")
        
    def map_solution_back(self, last_sol):
        """Converts P2 node values for flow back from column vector form, to 
        paired column (x,y)
        """
        sol = np.zeros(shape=(self.nodes.shape[0], 2))
        for node in self.P2_node_nums:
            x, y = self.node_dof_lookup_P2[node,:]
            
            sol[node,:] = np.array([last_sol[x], last_sol[y]])
            
        return sol
    def graph_domain(self):
        """Returns a simple graph of the domain triangulation and nodes"""
        
        vol_els = [a[1:4] for a  in self.els if a[0]==6 ]
        fig1 = plt.figure(figsize=(16,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_aspect('equal')
        
        tris = mtri.Triangulation(self.nodes[:,0],self.nodes[:,1],np.vstack(vol_els))
        ax1.triplot(tris, 'ko-', lw=1, ms=1.5)
        for i in range(1,15):
            dirichsx = [self.nodes[a,0] for a in self.dirich_nodes if self.dirich_nodes[a] == i]
            dirichsy = [self.nodes[a,1] for a in self.dirich_nodes if self.dirich_nodes[a] == i]
            if len(dirichsx) :
                ax1.plot(dirichsx,dirichsy, 'o',  label="Group"+str(i), ms=2.5)
            


        
        ax1.set_title('Domain')
        plt.tight_layout()
        plt.legend()
        return fig1

    def node_dof_mapping(self):
        """
        This allows for a rearrangement of the dofs not according to 
        the order of the gmsh node numbering.  In practice the ordering 
        here is exactly the gmsh node numbering. 
        """
        self.node_dof_lookup_P2 = -np.ones(shape=(self.nodes.shape[0],2), dtype=int)
        count = 0
        for node in self.P2_node_nums:
            self.node_dof_lookup_P2[node,0] = count
            self.node_dof_lookup_P2[node,1] = count + 1
            count += 2

        print(count, (self.nnodesP2)*2)
        
        
        count = 0
        # total of nodes because otherwise we need some complicated ordering
        self.node_dof_lookup_P1 = -np.ones(self.nodes.shape[0], dtype=int) 
        
        for node in self.P1_node_nums:
            self.node_dof_lookup_P1[node] = count
            count += 1
            
        print(count, (self.nnodesP1))
        if count!= (self.nnodesP1):
            raise ValueError("number of nodes allocated does not match number of P1 nodes")
            

    def get_element_node_coords(self, elnum:int):
        """Returns the coordinates of the element nodes as a numpy array shaped
        (nnodes, ndim)
        """
        if elnum < 0:
            raise ValueError("negative element numbers?")
            
        if elnum >= self.nels:
            raise ValueError("Element index out of range")
        
        elnodes = self.els[elnum,:]
        node_coords = np.zeros(shape=(elnodes[0], 2))
        for i in range(1,elnodes[0]+1):
            node_coords[i-1,:] = self.nodes[elnodes[i],:]
        
        return node_coords
        
    def get_element_global_dofs_P2(self, elnum:int):
        """Given an element number this method returns the P2 DOFs 
        for that element.
        """
        if elnum < 0:
            raise ValueError("negative element numbers?")
            
        if elnum >= self.nels:
            raise ValueError("Element index out of range")
            
        elnodes = self.els[elnum,:]
        glob_dofs = np.zeros(12, dtype=int)   
        for nod in range(elnodes[0]):
            glob_dofs[nod] = self.node_dof_lookup_P2[elnodes[1:][nod],0]
            glob_dofs[nod+6] = self.node_dof_lookup_P2[elnodes[1:][nod],1]
        return glob_dofs        

    def get_element_global_dofs_P1(self, elnum:int):
        """Given an element number this method returns the P1 DOFs 
        for that element.
        """
        if elnum < 0:
            raise ValueError("negative element numbers?")
            
        if elnum >= self.nels:
            raise ValueError("Element index out of range")
            
        elnodes = self.els[elnum,:4]
        glob_dofs = np.zeros(3, dtype=int)   
        for nod in range(3):
            glob_dofs[nod] = self.node_dof_lookup_P1[elnodes[1:][nod]]
        return glob_dofs
        
    def initial_value(self, vel:float):
        """Returns an initial velocity column vector 
        based on the boundary conditions.
        """
        nodes = self.nodes
        els = self.els
        uinit = np.zeros(self.nnodesP2*2)
        
        for node in self.inlet_nodes:
            i = self.node_dof_lookup_P2[node,0]
            uinit[i] = 1.
        for node in self.dirich_nodes:
            i,j = self.node_dof_lookup_P2[node,:]
            uinit[i] = 0.
            uinit[j] = 0.
            
        return uinit

    def determine_gauss_points(self, polydeg:int, ndim=2):
        """Determine the gauss points on the reference triangle for a legendre 
        polynomial of degree polydeg, dimensions are set to two for these meshes.
        
        returns the gauss points in rows with last column being the associated weight.
        """
        
        gauss_points_and_weights = pq.gauss_map_to_simplex(pq.gauss_points_box(polydeg, ndim))
        
        vol = gauss_points_and_weights[:,-1].sum()
        if abs(vol-0.5) > 1e6: # error check
            raise ValueError("Problem with gauss weights. They don't add up"+
                            " to the volume 0.5")
            
        self.gauss_points = gauss_points_and_weights.copy()
        

    def local_to_global_P1(self, argslist):

        """
        Local matrix is mapped with first all x than all y dofs. 
        Global alternates node1x node1y etc etc"""
        pshape = argslist[0][1]

        Pmat = scsp.lil_matrix(pshape)
        for k in range(len(argslist)):
            local, pshape, loc_nodes, Pnodes = argslist[k]
            for i in range(3):
                for j in range(12):
                    # could make this more efficient by recognising symmetry 
                    globi = Pnodes[i]
                    globj = loc_nodes[j]
                    Pmat[globi,globj] += local[i,j]
        return Pmat

    def local_to_global_P2(self, argslist):
        """
        Local matrix is mapped with first all x than all y dofs. 
        """
        Kshape = argslist[0][1]
        
        K = scsp.lil_matrix(Kshape)
        
        for k in range(len(argslist)):
            local, ks, loc_nodes = argslist[k]
            for i in range(12):
                for j in range(12):
                    # could make this more efficient by recognising symmetry 
                    globi = loc_nodes[i]
                    globj = loc_nodes[j]
                    K[globi,globj] += local[i,j]
        return K
        

    def convert_tempvec_to_flow(self, tmesh:ThermoMesh, tvec:np.array):
        """Used to input buouyancy to the forcing function
        """
        tdiffs_vec = np.zeros(self.nnodesP2*2)
        for tnode in tmesh.node_nums:
            tdof = tmesh.node_dof_lookup[tnode]
            fnode = tnode
            #fdofx = self.node_dof_lookup[fnode,0]
            fdofy = self.node_dof_lookup_P2[fnode,1]
            tdiffs_vec[fdofy] = tvec[tdof]
            
        return tdiffs_vec
        
    def reverse_element_ordering(self, elnum):        
        """Sometimes GMSH will produce an element with 
        backwards ordering. This will reverse the order.
        """
        old = self.els[elnum,:].copy()
        new = np.array([old[0],
                        old[1], old[3], old[2], 
                        old[6], old[5], old[4]])
        
        self.els[elnum,:] = new.copy()
        
    def add_surface_to_dirichlet(self, subdomains:tuple,  overwrite=False):
        """It is necessary to set some whole elements to zero flow sometimes.
        This method adds a given subdomain to the dirichlet set.
        WARNING: This method will cause singularities in the operators. You probably 
        want to remove the pressure terms for that element.
        """
        for l in subdomains:
            assert type(l) is int
        new_dc_nodes = []
        # get the elements of interest
        allelsofinterest = []
        for subdomain in subdomains:
            allelsofinterest.append([a for a in self.subdomains if self.subdomains[a] in subdomains])
 
        #get the nodes of interest
        allnodesofinterest = []
        for elgroup in allelsofinterest:
            nodegroup = []
            for el in elgroup:
                if self.els[el,0] ==6 :
                    nodes = self.els[el,1:7]
                    for i in range(6):
                        if nodes[i] not in new_dc_nodes:
                            nodegroup.append(nodes[i]) 
                else:
                    raise ValueError("Volume element "+str(el)+" is not a volume element")
                
            allnodesofinterest.append(nodegroup)
        
        #Remove pressure condition
        allelsconcat = [j for i in allelsofinterest for j in i]   
        newnnodesP1 = []
        for i in range(len(self.els)): 
            if self.els[i,0] ==6 and i not in allelsconcat: # add not yet added nodes to this list
                for node in self.els[i,1:4]:
                    if node not in newnnodesP1:
                        newnnodesP1.append(node) 
        self.P1_node_nums = newnnodesP1
        self.nnodesP1 = len(self.P1_node_nums)
        
        # add to dc nodes
        zipper = zip(subdomains,allnodesofinterest)
        for i in range(len(subdomains)):
            subdomain, new_dc_nodes = next(zipper)
            for node in new_dc_nodes:
                if (node not in self.dirich_nodes or overwrite) :
                    self.dirich_nodes[node] = subdomain 
                
        self.nnodes_dc = len(self.dirich_nodes)
        self.node_dof_mapping()
        
    def remove_dirichlet_group(self, group_tuple:tuple):
        """Will eliminate an edge from the dirichlet group, effectively converting
        to a Neumann boundary unless something else is done.
        """
        to_remove = []
        for node in self.dirich_nodes:
            if self.dirich_nodes[node] in (3,4):
                to_remove.append(node)
        print("Removing " + str(len(to_remove)) + " nodes")
        for node in to_remove:
            self.dirich_nodes.pop(node)
        self.nnodes_dc = len(self.dirich_nodes)
