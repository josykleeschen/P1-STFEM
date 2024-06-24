from Mesh import *
from ErrorEstimator import *
from Mark import *
from Solver import *
import numpy as np
import matplotlib.pyplot as plt
import time

# startmesh
grad_nodalbasis = np.array(([-1,-1],[1,0],[0,1]))
coordinates = np.array([(0,0),(1,0),(1,1),(0,1),(1/2,1/2)])
boundary = [[(0,1)],[(2,3)],[(1,2),(3,0)]]
elements = np.array([(0,1,4),(1,2,4),(2,3,4),(3,0,4)])

def f(var):
    return np.sin(np.pi*var[0])+np.sin(np.pi*var[0])*((var[1]*np.pi)**2)/2

def g(var):
    return 0

def v0(x):
    return 0

def s0(x):
    return 0

errorest = 2
tol = 0
start = time.process_time()
elementlist1 = []
timelist1 = []
errorestimate1 = []

#adaptive refinement
while len(elements) < 10**6 and errorest > tol:

    rco,rel,rbo = coordinates,elements,boundary
    nstart = time.process_time()
    print("Number of Elements: ", len(elements))
    elementlist1.append(len(elements))
    x, dellist = spsolve(elements, coordinates, boundary, grad_nodalbasis, f, g, s0, v0)
    n, errorest = estimate(elements, coordinates, grad_nodalbasis, f, g, s0, v0, dellist, x)
    print("Error Estimate: ", errorest)
    errorestimate1.append(errorest)
    marked = doerfler_marking(n, 1/4)
    if errorest > tol:
        coordinates, elements, boundary = nvb(coordinates, elements, boundary, marked)
    timelist1.append(time.process_time()-nstart)

end = time.process_time()

print("Time needed for execution of adaptive refinement:",end-start,"s")
for i in range(len(dellist)):
    x = np.insert(x, dellist[i], 0, 0)


# plot v_h
ax = plt.figure().add_subplot(projection = "3d")
ax.set_xlabel("x")
ax.set_ylabel("t")
tris = ax.plot_trisurf(rco[:,0],rco[:,1],x[:len(x)//2],cmap="cool")
plt.figure().colorbar(tris,ax = ax)
plt.show()

# plot sigma_h
ax = plt.figure().add_subplot(projection = "3d")
ax.set_xlabel("x")
ax.set_ylabel("t")
tris = ax.plot_trisurf(rco[:,0],rco[:,1],x[len(x)//2:],cmap="cool")
plt.figure().colorbar(tris,ax = ax)
plt.show()

# plot adaptive mesh if less than 5000 elements
if len(rel) < 5000:
    T = []
    for i in range(len(rel)):
        for j in range(3):
            T.append([rco[rel[i][j]][0],rco[rel[i][j]][1]])
    D = np.array(T)
    for i in range(len(T)//3):
        plt.gca().add_patch(plt.Polygon(D[i*3:i*3+3],edgecolor="black",facecolor="none"))
    plt.show()

coordinates = np.array([(0,0),(1,0),(1,1),(0,1),(1/2,1/2)])
boundary = [[(0,1)],[(2,3)],[(1,2),(3,0)]]
elements = np.array([(0,1,4),(1,2,4),(2,3,4),(3,0,4)])

tol = 0
errorest = 2
timelist2 = []
elementlist2 = []
errorestimate2 = []

#uniform refinement
while len(elements) < 1050000 and errorest > tol:
    rco,rel,rbo = coordinates,elements,boundary
    nstart = time.process_time()
    print("Number of Elements: ", len(elements))
    elementlist2.append(len(elements))
    x, dellist = spsolve(elements, coordinates, boundary, grad_nodalbasis, f, g, s0, v0)
    n, errorest = estimate(elements, coordinates, grad_nodalbasis, f, g, s0, v0, dellist, x)
    print("Error Estimate: ", errorest)
    errorestimate2.append(errorest)
    marked = np.arange(len(elements))
    if errorest > tol:
        coordinates, elements, boundary = nvb(coordinates, elements, boundary, marked)
    timelist2.append(time.process_time()-nstart)

# plot uniform mesh
if len(rel) < 5000:
    T = []
    for i in range(len(rel)):
        for j in range(3):
            T.append([rco[rel[i][j]][0],rco[rel[i][j]][1]])
    D = np.array(T)
    for i in range(len(T)//3):
        plt.gca().add_patch(plt.Polygon(D[i*3:i*3+3],edgecolor="black",facecolor="none"))
    plt.show()

# errorestimateplot
plt.loglog(elementlist1,errorestimate1,label="p = 1 adap.")
plt.loglog(elementlist1,np.array(elementlist1)**(-1/2),":",color = "black")
plt.loglog(elementlist2,errorestimate2,label="p = 1")
plt.xlabel("Number of Elements")
plt.ylabel("Error Estimate")
plt.legend(loc="best")
plt.show()

# timeplot
plt.loglog(elementlist1,timelist1,label="adaptive refinement")
plt.loglog(elementlist2,timelist2,label="uniform refinement")
plt.xlabel("Number of Elements")
plt.ylabel("Time in seconds")
plt.legend(loc="best")
plt.show()