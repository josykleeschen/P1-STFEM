import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def localAXX(det, InvJacobiTgradbasis):
    locAXX = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            locAXX[i, j] += (1 / 2) * det * (InvJacobiTgradbasis[j][1] * InvJacobiTgradbasis[i][1] + InvJacobiTgradbasis[j][0]*InvJacobiTgradbasis[i][0])

    return locAXX


def localAXY(det, InvJacobiTgradbasis):
    locAXY = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            locAXY[i, j] = (-1 / 2) * det * (InvJacobiTgradbasis[j][0] * InvJacobiTgradbasis[i][1] + InvJacobiTgradbasis[j][1]*InvJacobiTgradbasis[i][0])
    return locAXY


def localAYX(det, InvJacobiTgradbasis):
    return localAXY(det, InvJacobiTgradbasis).transpose()


def localAYY(det, InvJacobiTgradbasis):
    locAYY = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            locAYY[i, j] += (1 / 2) * det * (InvJacobiTgradbasis[j][0] * InvJacobiTgradbasis[i][0] + InvJacobiTgradbasis[j][1]*InvJacobiTgradbasis[i][1])

    return locAYY


def assemble_stiffness_matrix(elements, coordinates, lowerxbound, timebound, grad_nodal_basis):
    # Arrays needed to create Sparse Galerkin Matrix
    entries = np.empty(len(elements)*36+len(lowerxbound)*8)
    row_ind = np.empty(len(elements)*36+len(lowerxbound)*8)
    col_ind = np.empty(len(elements)*36+len(lowerxbound)*8)
    idxctr = 0

    for triangle in elements:
        # [x2-x1,x3-x1],[y2-y1,y3-y1]]
        detJacobi = -coordinates[triangle[1]][0] * coordinates[triangle[0]][1] + coordinates[triangle[2]][0] * \
                    coordinates[triangle[0]][1] + coordinates[triangle[0]][0] * coordinates[triangle[1]][1] - \
                    coordinates[triangle[2]][0] * coordinates[triangle[1]][1] - coordinates[triangle[0]][0] * \
                    coordinates[triangle[2]][1] + coordinates[triangle[1]][0] * coordinates[triangle[2]][1]
        InvJacobiT = (1 / -detJacobi) * np.array(([coordinates[triangle[0]][1] - coordinates[triangle[2]][1],
                                                   -coordinates[triangle[0]][0] + coordinates[triangle[2]][0]],
                                                  [-coordinates[triangle[0]][1] + coordinates[triangle[1]][1],
                                                   coordinates[triangle[0]][0] - coordinates[triangle[1]][
                                                       0]])).transpose()
        InvJacobiTgradbasis = np.empty((3, 2))
        for _ in range(3):
            InvJacobiTgradbasis[_] = np.dot(InvJacobiT,grad_nodal_basis[_].transpose())


        #AYX and AYY unnecessary since they are easily buildable using the other 2 local matrices
        locAXX = localAXX(detJacobi, InvJacobiTgradbasis)
        locAXY = localAXY(detJacobi, InvJacobiTgradbasis)

        for i in range(3):
            ridx = triangle[i]
            for j in range(3):
                cidx = triangle[j]
                entries[idxctr] = locAXX[i,j]
                entries[idxctr+1] = locAXY[i,j]
                entries[idxctr+2] = locAXY[j,i]
                entries[idxctr+3] = locAXX[i,j]
                row_ind[idxctr] = ridx
                row_ind[idxctr+1] = ridx+len(coordinates)
                row_ind[idxctr+2] = ridx
                row_ind[idxctr+3] = ridx+len(coordinates)
                col_ind[idxctr] = cidx
                col_ind[idxctr+1] = cidx
                col_ind[idxctr+2] = cidx+len(coordinates)
                col_ind[idxctr+3] = cidx+len(coordinates)
                idxctr += 4

    # add missing integral values
    for bound in lowerxbound:
        b0 = bound[0]
        b1 = bound[1]
        dist = np.abs(coordinates[bound[0]][0]-coordinates[bound[1]][0])
        entries[idxctr] = dist/3
        entries[idxctr+1] = dist/6
        entries[idxctr+2] = dist/6
        entries[idxctr+3] = dist/3
        entries[idxctr+4] = dist/3
        entries[idxctr+5] = dist/6
        entries[idxctr+6] = dist/6
        entries[idxctr+7] = dist/3
        row_ind[idxctr] = b0
        row_ind[idxctr+1] = b0
        row_ind[idxctr+2] = b1
        row_ind[idxctr+3] = b1
        row_ind[idxctr+4] = b0+len(coordinates)
        row_ind[idxctr+5] = b0+len(coordinates)
        row_ind[idxctr+6] = b1+len(coordinates)
        row_ind[idxctr+7] = b1+len(coordinates)
        col_ind[idxctr] = b0
        col_ind[idxctr+1] = b1
        col_ind[idxctr+2] = b0
        col_ind[idxctr+3] = b1
        col_ind[idxctr+4] = b0+len(coordinates)
        col_ind[idxctr+5] = b1+len(coordinates)
        col_ind[idxctr+6] = b0+len(coordinates)
        col_ind[idxctr+7] = b1+len(coordinates)
        idxctr += 8

    A = sp.csc_matrix((entries,(row_ind,col_ind)),shape=(2*len(coordinates),2*len(coordinates)))
    # delete superfluous rows and columns -> where basis is 0
    dellist = np.unique(timebound)
    mask = np.ones(A.shape[0], dtype=bool)
    mask[dellist] = False
    A = A[:,mask]
    A = A[mask,:]

    return A, dellist

def localFX(det, z, InvJacobiTgradbasis, f, g):

    locfX = (1/2)*det*(f((1/3)*(z[0]+z[1]+z[2]))*(InvJacobiTgradbasis[:,1])-g((1/3)*(z[0]+z[1]+z[2]))*(InvJacobiTgradbasis[:,0]))
    return locfX

def localFY(det, z, InvJacobiTgradbasis, f, g):

    locfY = (1/2)*det*(-f((1/3)*(z[0]+z[1]+z[2]))*(InvJacobiTgradbasis[:,0])+g((1/3)*(z[0]+z[1]+z[2]))*(InvJacobiTgradbasis[:,1]))
    return locfY

def assemble_rhs(elements, coordinates, lowerxbound, timebound, grad_nodal_basis, f, g, s0, v0):
    b = np.zeros(2*len(coordinates))
    for triangle in elements:
        detJacobi = -coordinates[triangle[1]][0] * coordinates[triangle[0]][1] + coordinates[triangle[2]][0] * \
                    coordinates[triangle[0]][1] + coordinates[triangle[0]][0] * coordinates[triangle[1]][1] - \
                    coordinates[triangle[2]][0] * coordinates[triangle[1]][1] - coordinates[triangle[0]][0] * \
                    coordinates[triangle[2]][1] + coordinates[triangle[1]][0] * coordinates[triangle[2]][1]
        InvJacobiT = (1/-detJacobi)*np.array(([coordinates[triangle[0]][1]-coordinates[triangle[2]][1], -coordinates[triangle[0]][0]+coordinates[triangle[2]][0]],[-coordinates[triangle[0]][1]+coordinates[triangle[1]][1],coordinates[triangle[0]][0]-coordinates[triangle[1]][0]])).transpose()
        InvJacobiTgradbasis = np.empty((3, 2))
        z = np.array((coordinates[triangle[0]],coordinates[triangle[1]],coordinates[triangle[2]]))

        for _ in range(3):
            InvJacobiTgradbasis[_] = np.dot(InvJacobiT,grad_nodal_basis[_].transpose())

        locfX = localFX(detJacobi,z,InvJacobiTgradbasis,f,g)
        locfY = localFY(detJacobi,z,InvJacobiTgradbasis,f,g)

        b[triangle] += locfX
        b[triangle+len(coordinates)] += locfY

    # add missing integral values
    for bound in lowerxbound:
        dist = np.abs(coordinates[bound[0]][0] - coordinates[bound[1]][0])
        b[bound] += (dist/2) * v0((1/2) * (coordinates[bound[0]][0] + coordinates[bound[1]][0]))
        b[bound+len(coordinates)] += (dist/2) * s0((1/2) * (coordinates[bound[0]][0] + coordinates[bound[1]][0]))

    # delete superfluous rows where basis is 0.
    dellist = np.unique(timebound)
    mask = np.ones(b.shape,dtype=bool)
    mask[dellist] = False

    return b[mask]

def spsolve(elements, coordinates, boundary, grad_nodal_basis, f, g, s0, v0):
    A, dellist = assemble_stiffness_matrix(elements, coordinates, np.array(boundary[0]), np.array(boundary[2]), grad_nodal_basis)
    b = assemble_rhs(elements, coordinates, np.array(boundary[0]), np.array(boundary[2]), grad_nodal_basis, f, g, s0, v0)
    x = sp.linalg.spsolve(A,b)
    return x, dellist