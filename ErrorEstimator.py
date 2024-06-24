import numpy as np

def vh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,index):
    sum = 0
    for i in range(3):
        sum += sol[triangle[i]] * (InvJacobiT@grad_nodal_basis[i].transpose())[index]
    return sum

def sh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,index):
    sum = 0
    for i in range(3):
        sum += sol[triangle[i]+len(sol)//2] * (InvJacobiT@grad_nodal_basis[i].transpose())[index]
    return sum

def estimate(elements, coordinates, grad_nodal_basis, f, g, s0, v0, dellist, sol):
    for i in range(len(dellist)):
        sol = np.insert(sol, dellist[i], 0, 0)
    errorlist = np.empty(len(elements))
    idxctr = 0
    for triangle in elements:
        nT = 0
        nodes = [coordinates[triangle[0]],coordinates[triangle[1]],coordinates[triangle[2]]]
        detJacobi = -coordinates[triangle[1]][0] * coordinates[triangle[0]][1] + coordinates[triangle[2]][0] * \
                    coordinates[triangle[0]][1] + coordinates[triangle[0]][0] * coordinates[triangle[1]][1] - \
                    coordinates[triangle[2]][0] * coordinates[triangle[1]][1] - coordinates[triangle[0]][0] * \
                    coordinates[triangle[2]][1] + coordinates[triangle[1]][0] * coordinates[triangle[2]][1]
        InvJacobiT = (1 / -detJacobi) * np.array(([coordinates[triangle[0]][1] - coordinates[triangle[2]][1],
                                                   -coordinates[triangle[0]][0] + coordinates[triangle[2]][0]],
                                                  [-coordinates[triangle[0]][1] + coordinates[triangle[1]][1],
                                                   coordinates[triangle[0]][0] - coordinates[triangle[1]][
                                                       0]])).transpose()
        nT += ((detJacobi/2)*(vh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,1)-sh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,0)-f((1/3)*(nodes[0]+nodes[1]+nodes[2])))**2 +
               (detJacobi/2)*(sh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,1)-vh_sum(sol,grad_nodal_basis,InvJacobiT,triangle,0)-g((1/3)*(nodes[0]+nodes[1]+nodes[2])))**2)
        for i in range(3):
            if nodes[i][1] == 0 and nodes[(i + 1) % 3][1] == 0:
                nT += np.abs(nodes[i][0] - nodes[(i + 1) % 3][0]) * (((1/2) * (sol[triangle[i]]+sol[triangle[(i+1)%3]]) - v0((1/2) * (nodes[i][0] + nodes[(i + 1) % 3][0]))) ** 2 + ((1/2) * (sol[triangle[i]+len(coordinates)]+sol[triangle[(i+1)%3]+len(coordinates)]) - s0((1/2) * (nodes[i][0] + nodes[(i + 1) % 3][0]))) ** 2)
        errorlist[idxctr] = nT
        idxctr += 1
    errorest = np.sqrt(np.sum(errorlist))
    return errorlist, errorest