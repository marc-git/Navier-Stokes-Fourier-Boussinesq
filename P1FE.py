#/bin/python3

# Not much in here as we don't need derivatives of P1 shape functions
# when they are used only for pressure.

import numpy as np

    
    
def shape_func_vec_P1(ref_point):
    """Input should be a numpy vector of length 2.  Range should be
    inside the unit reference triangle.
    
    Returns a numpy vector of length three representing the node
    values of the P1 shape functions at the given reference point.
    
    """
    
    node_vals = np.zeros(3)
    node_vals[0] = 1.-sum(ref_point)
    node_vals[1] = ref_point[0]
    node_vals[2] = ref_point[1]
    
    return node_vals
