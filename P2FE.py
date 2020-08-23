#/bin/python3


#Numerics
import numpy as np


"""
---------------------------------------------------------------------------------
| Node function stuff                                                           |
---------------------------------------------------------------------------------
"""
        
def shape_func(ref_point, node_num : int):
    """Returns the value of the specific shape function
    
    Input : xi, eta local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
                    
            node_num the number of the node as per VTK element 22 numbered 1-6
            
    """
    xi, eta = ref_point
    result=None
    if not (1<=node_num<=6):
        raise ValueError("Node Numbers are 1 to 6")
        
    if not(0.<=xi<=1.) or not(0.<=eta<=1.) or (abs(xi)+abs(eta)>1.0):
        raise ValueError( "Not inside reference triangle")
        

    
    if node_num==1:
        result = (eta+xi-1.)*(2*eta+2*xi-1.) # correct
    
    
    if node_num==2:
        result = xi* (2*xi-1.)        # correct
    
    
    if node_num==3:
        result = eta*(2*eta-1.)       # correct
    
    
    if node_num==4:
        result = -4*xi*(eta+xi-1.)    # correct
    
    
    if node_num==5:
        result = 4*eta*xi             # correct
    
    
    if node_num==6:
        result =  -4*eta*(eta+xi-1.)  # correct
    
    return result

def shape_func_vec(ref_point):
    """Returns the value of the specific shape function
    
    Input : xi, eta local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
                    
            node_num the number of the node as per VTK element 22 numbered 1-6
            
    """
    xi, eta = ref_point
    result=None
    
    if not(0.<=xi<=1.) or not(0.<=eta<=1.) or (abs(xi)+abs(eta)>1.0):
        raise ValueError( "Not inside reference triangle")
        

    result = np.zeros(6)
    result[0] = (eta+xi-1.)*(2*eta+2*xi-1.) # correct
    result[1] = xi* (2*xi-1.)        # correct
    result[2] = eta*(2*eta-1.)       # correct
    result[3] = -4*xi*(eta+xi-1.)    # correct
    result[4] = 4*eta*xi             # correct
    result[5] = -4*eta*(eta+xi-1.)   # correct
    
    return result
    
def shape_func_deriv(ref_point, node:int, deriv_num):
    """Returns the value of the derivative of the node shape function
    
    
    Input : xi, eta local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
                    
            node_num the number of the node as per VTK element 22 numbered 1-6
            
            deriv_num : between 1,2 determines whether we are derivating by xi or eta
    """
    
    node_num = node
    xi, eta = ref_point
    result=None
    if not (1<=node_num<=6):
        raise ValueError("Node Numbers are 1 to 6")
        
    if not(0.<=xi<=1.) or not(0.<=eta<=1.) or (abs(xi)+abs(eta)>1.0):
        raise ValueError("Not inside reference triangle")
        
    if deriv_num!=1 and deriv_num!=2:
        raise ValueError( "Deriv_num must be one or two, not "+str(deriv_num))
        
    
    if deriv_num==1:

        if node_num==1:
            result =  (4*xi+4*eta-3.) # correct


        if node_num==2:
            result =    (4*xi-1.)    # correct


        if node_num==3:
            result = 0.    # correct


        if node_num==4:
            result = -4*(2*xi+eta-1)   # correct


        if node_num==5:
            result = 4*eta # correct


        if node_num==6:
            result = -4*eta  # correct

    else:
        

        if node_num==1:
            result = (4*xi+4*eta-3.)  # correct


        if node_num==2:
            result = 0.   # correct


        if node_num==3:
            result = (4*eta-1.)   # correct


        if node_num==4:
            result = -4*xi    # correct


        if node_num==5:
            result = 4*xi   # correct


        if node_num==6:
            result =  -4*(xi+2*eta-1)    # correct
            
    return result

def shape_func_deriv_vec(ref_point):
    """Returns the value of the derivative of the node shape function
    
    
    Input : xi, eta local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
                    
            node_num the number of the node as per VTK element 22 numbered 1-6
            
            deriv_num : between 1,2 determines whether we are derivating by xi or eta
    """
    
    
    xi, eta = ref_point
    
    
        
    if not(0.<=xi<=1.) or not(0.<=eta<=1.) or (abs(xi)+abs(eta)>1.0):
        raise ValueError("Not inside reference triangle")
        
        
    result = np.zeros(shape=(6,2))
    
    # xi deriv
    result[0,0] =  (4*xi+4*eta-3.) # correct
    result[1,0] =    (4*xi-1.)     # correct
    result[2,0] = 0.               # correct
    result[3,0] = -4*(2*xi+eta-1)  # correct
    result[4,0] = 4*eta            # correct
    result[5,0] = -4*eta           # correct

    #eta deriv
    result[0,1] = (4*xi+4*eta-3.)  # correct
    result[1,1] = 0.               # correct
    result[2,1] = (4*eta-1.)       # correct
    result[3,1] = -4*xi            # correct
    result[4,1] = 4*xi             # correct
    result[5,1] =  -4*(xi+2*eta-1) # correct
            
    return result
    
def local_node_derivs(xi_eta, node_coords):
    """Get gradient in local coords, not in reference for the purpose of building the stiffness matrix
    
    """
    jac = jacobian(xi_eta, node_coords)
    if np.linalg.det(jac)!= 0:
        jacinv = np.linalg.inv(jac)
    else:
        print(node_coords)
        raise ValueError("Jacobian has zero determinant")
    
    phi_derivs = shape_func_deriv_vec(xi_eta)
    phi_local_deriv = jacinv.dot(phi_derivs.transpose()) # 2x6 matrix
    
    
    return phi_local_deriv.transpose()
    
    

"""
---------------------------------------------------------------------------------
| Mapping function stuff                                                        |
---------------------------------------------------------------------------------
"""

def map_to_local(xi_eta, coeffs):
    """Map any reference element coords to the physical element
    
    Input:  xi_eta: local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
            
            coeffs: numpy array with shape (6,2) where node_coords[0,0] is the 
                            x coord of the first node
                            
    
    Output the x,y coords as numpy array
            
    """
    xi, eta = xi_eta
    c1,c2,c3,c4,c5,c6 = coeffs
    res = (2*eta**2*(c1+c3-2*c6) + 
          eta*xi*4*(c1-c4+c5-c6)+
          eta*(4*c6-3*c1-c3)+
          2*xi**2*(c1+c2-2*c4)+
          xi*(4*c4-c2-3*c1)+
          c1)
    
    return res
    
    
def jacobian(xi_eta, node_coords):
    """Returns the jacobian for the given xi_eta points
        
    Input:  xi_eta: local coords must be inside a reference triangle with vertices 
                    (0,0), (1,0), (0,1)
            
            node_coords: numpy array with shape (6,2) where node_coords[0,0] is the 
                            x coord of the first node
                            
    """
    
    jac = np.zeros(shape=(2,2))
    for i in range(6):
        nx = shape_func_deriv(xi_eta, i+1, 1)
        ny = shape_func_deriv(xi_eta, i+1, 2)
        jac[0,0] += nx*node_coords[i,0]
        jac[0,1] += nx*node_coords[i,1]
        jac[1,0] += ny*node_coords[i,0]
        jac[1,1] += ny*node_coords[i,1]
    
    return jac
    
"""
---------------------------------------------------------------------------------
| Buoyancy Calculations                                                         |
---------------------------------------------------------------------------------
"""

def local_forcing_el( el_node_coords:np.array, int_points:np.array):
    """This is used to calculate the buoyancy. Will be multiplied later by
    the temperature and expansion coefficient.
    
    Input should be the coordinate of the element nodes (6),
    as well as the gaussian integration points, which should be in format
    (x, y, weight)
    
    Returns the integral of the shape functions over the element for the
    indicator function.
    """
    

    npoints = int_points.shape[0]
    
    el_vec = np.zeros(6)
    for i in range(npoints):
        det = np.linalg.det(jacobian(int_points[i,:2], el_node_coords))
        el_vec += (shape_func_vec(int_points[i,:2])*int_points[i,-1] *det)
        if det<0.:
            print("Negative determinant in element!" )
    return el_vec



