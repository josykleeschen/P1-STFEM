import numpy as np
import scipy.sparse as sp


# function that returns the necessary geometric data to implement the Newest Vertex Bisection (nvb) Adaptive Mesh Refinement method
def geometricdata(elements, bound):
    nE = len(elements)
    nB = len(bound)
    I = np.zeros(3 * len(elements) + len(bound[0]) + len(bound[1]) + len(bound[2]), dtype=int)
    J = np.zeros(3 * len(elements) + len(bound[0]) + len(bound[1]) + len(bound[2]), dtype=int)
    I[:3*len(elements)] = elements.flatten(order="F")
    J[:3*len(elements)] = elements[:,[1,2,0]].flatten(order="F")
    pointer = [3*nE]
    ctr = 0
    for j in range(nB):
        for l in range(len(bound[j])):
            I[ctr+3*len(elements)] = bound[j][l][1]
            J[ctr+3*len(elements)] = bound[j][l][0]
            ctr+=1
        pointer.append(pointer[j] + len(bound[j]))
    idxIJ = np.argwhere(I < J).ravel()
    idxJI = np.argwhere(J < I).ravel()
    edgenumber = np.zeros([I.size, 1])
    i = 1
    for idx in idxIJ:
        edgenumber[idx] = i
        i += 1
    number2edges = sp.csc_matrix((np.arange(len(idxIJ)) + 1, (I[idxIJ], J[idxIJ])))
    numberingIJ = number2edges.data
    JI2IJ = sp.csc_matrix((idxJI, (J[idxJI], I[idxJI])))
    idxJI2IJ = JI2IJ.data
    j = 0
    for idx in idxJI2IJ:
        edgenumber[idx] = numberingIJ[j]
        j += 1
    boundary2edges = []
    for j in range(nB):
        boundary2edges.append(edgenumber[pointer[j]:pointer[j+1]].astype(int) -1)
    element2edges = np.reshape(edgenumber[:3 * nE], [nE, 3], order="F").astype(int) -1
    edge2nodes = np.array([I[idxIJ], J[idxIJ]]).transpose()
    return edge2nodes, element2edges, boundary2edges


# function that performs a single iteration of the nvb method using given coordinates, elements, boundary edges and an array of elements that should be refined
# returns the new mesh (-> new coordinates, new elements and new boundary edges)
def nvb(coords, elements, bound, marked):
    if marked.size == 0:
        return coords, elements, bound
    boundary = bound
    edge2nodes, element2edges, boundary2edges = geometricdata(elements,bound)
    edge2newNode = np.zeros(np.max(element2edges,initial=0) + 1)
    edge2newNode[element2edges[marked]] = 1

    # This part is to make sure that the opposite edge is also marked for refinement
    swap = np.array([0])
    while swap.size != 0:
        markedEdge = edge2newNode[element2edges]
        swap = np.argwhere(markedEdge[:, 0] == 0)
        swap = swap[np.argwhere(markedEdge[swap, 1] + markedEdge[swap, 2] >= 1)[:,0]]
        edge2newNode[element2edges[swap, 0]] = 1

    # Here we create the new coordinates
    edge2newNode[edge2newNode.nonzero()] = len(coords) + np.arange(edge2newNode.nonzero()[0].size)
    edge2newNode = edge2newNode.astype(int)
    idx = np.transpose(edge2newNode.nonzero())
    coords = np.concatenate((coords, np.zeros((idx.shape[0], 2))), axis=0)
    coords[edge2newNode[idx]] = (coords[edge2nodes[idx, 0],:] + coords[edge2nodes[idx, 1],:]) / 2

    # This part updates the boundary
    refboundary = bound.copy()
    for j in range(len(bound)):
        if len(boundary[j]) != 0:
            tmp = np.array(boundary[j])
            newNodes = edge2newNode[boundary2edges[j]].flatten(order="F")
            markedEdges = newNodes.nonzero()[0]
            if markedEdges.size != 0:
                refboundary[j] = np.concatenate((tmp[np.squeeze(np.argwhere(newNodes == 0),axis=1)],
                                 np.concatenate(([tmp[markedEdges, 0]], [newNodes[markedEdges]]),axis=0).transpose(),
                                 np.concatenate(([newNodes[markedEdges]], [tmp[markedEdges, 1]]),axis=0).transpose()), axis=0)

    # This part sorts the cases of refinement we use on the respective element
    newNodes = edge2newNode[element2edges]
    none = np.array([1 if newNodes[i, 0] == 0 else 0 for i in range(len(newNodes))])
    bisec1 = np.array([1 if newNodes[i, 0] != 0 and newNodes[i, 1] == 0 and newNodes[i, 2] == 0 else 0 for i in range(len(newNodes))])
    bisec12 = np.array([1 if newNodes[i, 0] != 0 and newNodes[i, 1] != 0 and newNodes[i, 2] == 0 else 0 for i in range(len(newNodes))])
    bisec13 = np.array([1 if newNodes[i, 0] != 0 and newNodes[i, 1] == 0 and newNodes[i, 2] != 0 else 0 for i in range(len(newNodes))])
    bisec123 = np.array([1 if newNodes[i, 0] != 0 and newNodes[i, 1] != 0 and newNodes[i, 2] != 0 else 0 for i in range(len(newNodes))])
    idx = none + bisec1 * 2 + bisec12 * 3 + bisec13 * 3 + bisec123 * 4
    idx = np.hstack(([0], np.cumsum(idx)))
    newElements = np.zeros((idx[-1], 3),int)

    # This part puts the 4 (5, if doing nothing counts) different refinement methods into action
    newElements[idx[none.nonzero()]] = elements[none.nonzero()]  # do not refine
    bn1 = bisec1.nonzero()[0]
    if bn1.size != 0:
        # refine edge 1
        newElements[np.hstack((idx[bn1], 1 + idx[bn1]))] = np.hstack((np.vstack(
            (elements[bn1, 2], elements[bn1, 0], newNodes[bn1, 0])), np.vstack(
            (elements[bn1, 1], elements[bn1, 2], newNodes[bn1, 0])))).transpose()
    bn12 = bisec12.nonzero()[0]
    if bn12.size != 0:
        # refine edge 1 and 2
        newElements[np.hstack((idx[bn12], 1 + idx[bn12], 2 + idx[bn12]))] = np.hstack((np.vstack(
            (elements[bn12, 2], elements[bn12, 0], newNodes[bn12, 0])), np.vstack(
            (newNodes[bn12, 0], elements[bn12, 1], newNodes[bn12, 1])), np.vstack(
            (elements[bn12, 2], newNodes[bn12, 0], newNodes[bn12, 1])))).transpose()
    bn13 = bisec13.nonzero()[0]
    if bn13.size != 0:
        # refine edge 1 and 3
        newElements[np.hstack((idx[bn13], 1 + idx[bn13], 2 + idx[bn13]))] = np.hstack((np.vstack(
            (newNodes[bn13, 0], elements[bn13, 2], newNodes[bn13, 2])), np.vstack(
            (elements[bn13, 0], newNodes[bn13, 0], newNodes[bn13, 2])), np.vstack(
            (elements[bn13, 1], elements[bn13, 2], newNodes[bn13, 0])))).transpose()
    bn123 = bisec123.nonzero()[0]
    if bn123.size != 0:
        # refine all edges
        newElements[np.hstack((idx[bn123], 1 + idx[bn123], 2 + idx[bn123], 3 + idx[bn123]))] = np.hstack((np.vstack(
            (newNodes[bn123, 0], elements[bn123, 2], newNodes[bn123, 2])), np.vstack(
            (elements[bn123, 0], newNodes[bn123, 0], newNodes[bn123, 2])), np.vstack(
            (newNodes[bn123, 0], elements[bn123, 1], newNodes[bn123, 1])), np.vstack(
            (elements[bn123, 2], newNodes[bn123, 0], newNodes[bn123, 1])))).transpose()
    return coords, newElements, refboundary
