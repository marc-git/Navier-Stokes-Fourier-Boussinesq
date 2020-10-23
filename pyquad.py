#/bin/python3

"""
Version 2020-06

The below code calculates the Gauss-Legendre quadrature points to arbitrary degree. 
No limit is imposed on the number of dimensions. 

Disclaimer: If you rely on the accuracy of this for real world problems then do your own testing.
 This is without warranty. The only check I did was to make sure the determinants add to the 
 n-volume of the integration domain. You can easily do this too. 

Copyright 2020, Marc Graham

License: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import math
from itertools import combinations, product
import numpy as np



def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def legendre_poly(n:int,x:float):
    # calculate Legendre Polynomial using the Rodrigues derivation

    p = 1.
    if n<0 :
        raise ValueError("negative polynomial order")
    if -1.>x or 1.<x:
        raise ValueError("Only for range x in [-1, 1]")
    if n>0:
        p = 0.
        for k in range(0,n+1):
            p += binom(n,k)*binom(n+k,k)*((x-1.)/2)**k
    return p
    

def legendre_poly_diff(n:int, x:float):
    # calculate the derivative using the differential rule
    if n<0 :
        raise ValueError("negative polynomial order")
    if n==0:
        return 0.
    if -1.>=x or 1.<=x:
        raise ValueError("Only for range x in (-1, 1)")
    dpdx = (x*legendre_poly(n,x)-legendre_poly(n-1,x))*n/(x**2-1)
    return dpdx
    

def calculate_legendre_roots(n:int):
    """Solves the roots of the legendre polynomial up to 
    order n by determining eigenvalues of the jacobi matrix.
    Weights are calculated using the common rule for the 
    standard interval.
    """
    if n<0 :
        raise ValueError("negative polynomial order")
    if n==0:
        return None
    
    A = np.zeros(shape=(n,n))
    for i in range(1,n):
        A[i-1,i] = i/np.sqrt(4*i**2-1)
    A += A.transpose()
    roots, vecs = np.linalg.eigh(A)
    weights = np.zeros_like(roots)
    for i in range(roots.shape[0]):
        x = roots[i]
        weights[i] = 2./(legendre_poly_diff(n,x)**2*(1-x**2))
    return roots, weights
    
    
def vec_to_grid(vec:np.array, weight:np.array, ndim:int):
    """ Does the cartesian product of points to get grid of integration points
    """
    l = vec.shape[0]
    if len(vec.shape)!=1:
        raise IndexError("This vec should be a vector")
    
    # prepare a vector with the mapped point and the associated weight
    out = np.zeros(shape=(l**ndim,ndim+1))
    
    comb = product(list(range(l)),repeat=ndim) #itertools.product
    i = 0
    for p in comb:
        out[i,ndim] = 1.
        for d in range(ndim):
            x = vec[p[d]]
            out[i,d] = x
            out[i,ndim] *= weight[p[d]]
        i += 1
    return out
    
"""

***********************************************************************
*
* Primary function here!
*
***********************************************************************
"""    

def gauss_points_box(npoly:int, ndim:int):
    """Produces the gaussian integration points and weights (last column)
    corresponding to the Legendre-Polynomial of degree npoly, in ndim. 
    Note that the ndim works for ndim>3 but didn't really think about 
    how right this is. """
    r,w = calculate_legendre_roots(npoly)
    out = vec_to_grid(r,w,ndim)
    
    return out
    
    

"""

***********************************************************************
*
* The remainder is for mapping to the d-simplex
*
***********************************************************************
"""    
def build_mapping(ndim:int):
    """Builds the mapping from the line/square/cube [-1,1]^d to the reference triangle
    which has vertices at {e_i} for 1<= i <= d
    """
    if ndim<=0 :
        raise ValueError("minimum dimension is obviously one")
    
    quad_vert = np.zeros(shape=(2**ndim,ndim))

    for i in range(2**(ndim-1)):
        quad_vert[0+2*i,0] = -1.*(-1)**i
        quad_vert[1+2*i,0] = 1.*(-1)**i

    for col in range(1, ndim):

        vec = -np.ones(2**(col))
        vec = np.hstack((vec,-vec))
        l = vec.shape[0]
        for i in range(2**(ndim-col-1)):
            quad_vert[l*i:l*i+l,col] = vec
    
    # simplex vertices
    vertices = np.zeros(shape=(ndim+1,ndim))
    for i in range(1,ndim+1):
        vertices[i,i-1] = 1.

    #mapping 
    # get every combination of first order variables
    pterms = [(0,)]
    directions = list(range(1,ndim+1))
    for i in range(1, ndim+1):
        pterms += list(combinations(directions,i))
    l = len(pterms)
    
    # calculate the factor matrix and invert
    M = np.ones(shape=(l,l))
    for i in range(quad_vert.shape[0]):
        M[i,0] = 1.
        point = quad_vert[i,:]
        for j in range(1,l):
            v = pterms[j]
            for k in v:
                M[i,j] *= point[k-1]
        
    # now we can build the shape functions
    Minv = np.linalg.inv(M)
    
    mapto = np.zeros_like(quad_vert)
    # first move all the points in the (1,1,1,1) direction
    # then shrink by two
    for i in range(quad_vert.shape[0]):
        mapto[i,:] = (quad_vert[i,:]+1.)/2
        
    # if any point coincides with a simplex vertex
    # then it does not move
    # for the others, if the point is equidistant
    # from two or more other points, then it goes to an average
    # of those points
    eps = 1e-6
    for i in range(quad_vert.shape[0]):
        point = mapto[i,:]
        dist = np.zeros(ndim+1)
        for j in range(ndim+1):
            dist[j] = np.linalg.norm(point-vertices[j])
        mindist = dist.min()
        lowdistgroup = [vertices[k] for k in range(ndim+1) 
                        if np.linalg.norm(point-vertices[k]) <= mindist+eps]
        mapto[i,:] = sum(lowdistgroup)/len(lowdistgroup)
    
    return pterms, mapto, Minv
    

def pterms_diff(pterm, diffdir):
    """
    returns the differential of a term above in the direction diffdir"""
    if diffdir in pterm:
        still_there = [a for a in pterm if a!=diffdir]
        if len(still_there)==0:
            still_there = 1.
        else:
            still_there = tuple(still_there)
        return (still_there)
    else:
        return 0.
    
def apply_map(coord:np.array, pterms:list, mapto:np.array, Minv:np.array):
    """Applies the built map to a point
    """
    
    if len(coord.shape)!=1:
        raise IndexError("This coord should be a vector")
    
    ndim = coord.shape[0]
    
    if len(pterms)!=2**ndim:
        raise IndexError("not enough poly terms in pterms")
    if mapto.shape[0]!=2**ndim:
        raise IndexError("Not enough mapping points to determine system")
    if Minv.shape[0]!=Minv.shape[1] or Minv.shape[0]!=2**ndim:
        raise IndexError("Minv has incompatible dimensions")
    
    ptvals = np.ones(2**ndim)
    
    for i in range(1,len(pterms)):
        for d in pterms[i]:
            ptvals[i] *= coord[d-1]
    shapes = ptvals.dot(Minv).transpose()
    res = np.zeros(ndim)
    for i in range(ndim):
        res[i] = sum([mapto[j,i]*shapes[j] for j in range(2**ndim)])
    
    return res
    
    
def jacobian(coord:np.array, pterms:list, mapto:np.array, Minv:np.array):
    
    if len(coord.shape)!=1:
        raise IndexError("This coord should be a vector")
    
    ndim = coord.shape[0]
    
    if len(pterms)!=2**ndim:
        raise IndexError("not enough poly terms in pterms")
    if mapto.shape[0]!=2**ndim:
        raise IndexError("Not enough mapping points to determine system")
    if Minv.shape[0]!=Minv.shape[1] or Minv.shape[0]!=2**ndim:
        raise IndexError("Minv has incompatible dimensions")
    
    # calculate already the derivatives for the jacobian
    diffs = []
    l = len(pterms)
    for d in range(1,ndim+1):
        ptd = []
        for i in range(l):
            ptd.append(pterms_diff(pterms[i], d))
        diffs.append(ptd)
    
    jac = np.zeros(shape=(ndim,ndim))
    
    for j in range(ndim):
        dif = diffs[j]
        difvec = np.ones(2**ndim)
        for k in range(2**ndim):
            if type(dif[k]) is float:
                difvec[k] = dif[k]
            elif type(dif[k]) is tuple:
                for d in dif[k]:
                    difvec[k] *= coord[d-1]
            else:
                raise TypeError("only floats and tuples allowed here")
        dshapes = difvec.dot(Minv).transpose()
        for i in range(ndim):
            jac[i,j] = sum([mapto[k,i]*dshapes[k] for k in range(2**ndim)])
            
    return jac

"""

***********************************************************************
*
* Primary function here!
*
***********************************************************************
"""    

    
    
def gauss_map_to_simplex(g):
    ndim = g.shape[1]-1
    ps = g.shape[0]
    pterms, mapto, Minv = build_mapping(ndim)
    g_mapped = g.copy()
    
    for i in range(ps):
        p = g[i,:ndim]
        g_mapped[i,:ndim] = apply_map(p, pterms, mapto, Minv)
        g_mapped[i,ndim] *= np.linalg.det(jacobian(p, pterms, mapto, Minv))
        
    return g_mapped
        

    
