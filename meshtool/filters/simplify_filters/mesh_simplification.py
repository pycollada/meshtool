import numpy
from numpy import array
import copy
import heapq
import math
from .progress_printer import ProgressPrinter

def quadricForTriangle(triangle, progress):
    progress.step()
    a,b,c = triangle
    normal = numpy.cross(b - a, c - a)
    normal = normal / numpy.linalg.norm(normal)
    s1 = numpy.linalg.norm(b-a)
    s2 = numpy.linalg.norm(c-a)
    s3 = numpy.linalg.norm(c-b)
    sp = (s1+s2+s3)/2.0
    area2 = sp*(sp-s1)*(sp-s2)*(sp-s3)
    if area2 <= 0:
        area = 0 # Floating point error can sometimes cause this
    else:
        area = math.sqrt(area2)
    d = -numpy.dot(normal, a)
    return (area*numpy.outer(normal, normal), area*d*normal, area*d*d)
    

def evalQuadric(A, b, c, pt):
    return numpy.dot(pt,numpy.inner(A,pt)) + 2*numpy.dot(b,pt) + c

class MeshSimplification:

    # Instance variables:
    #
    # vertices: array of vertices
    #
    # triangles: array of triangles (each triangle:
    # 3 indices into vertices array)
    #
    # NO ____  adj: array of lists of triangle indices that a vertex appears on (parallel
    # to vertices array)
    #
    # quadrics: quadric of each vertex (parallel to vertices array)
    #
    # contractions: array of lists of contractions that a vertex appears in

    def __init__(self, vertices, triangles, corner_attributes=[]):
        print "Copying..."
        self.vertices = [copy.copy(vertex) for vertex in vertices]
        self.triangles = [copy.copy(triangle) for triangle in triangles]
        self.corner_attributes = [[copy.copy(triangle) for triangle in attr_list]
                                  for attr_list in corner_attributes]
        self.adj = [{} for i in range(len(vertices))]
        print "Building simplex..."
        progress = ProgressPrinter(len(triangles))
        for i in range(len(triangles)):
            progress.step()
            (a,b,c) = triangles[i]
            self.adj[a][i] = 0
            self.adj[b][i] = 1
            self.adj[c][i] = 2
        print "Generating triangle quadrics..."
        progress = ProgressPrinter(len(triangles))
        self.tri_quadrics = [quadricForTriangle(triangle, progress)
                             for triangle in vertices[triangles]]
        print "Computing vertex quadrics..."
        progress = ProgressPrinter(len(vertices))
        self.quadrics = [self.vertexQuadric(i, progress) for i in range(len(vertices))]
        del self.tri_quadrics # Not needed anymore
        self.heap = []
        self.numContr = 0
        self.contractions = [{} for i in range(len(vertices))]
        self.contractionsByVertices = {}
        print "Generating contractions..."
        progress = ProgressPrinter(len(triangles))
        for i in range(len(triangles)):
            progress.step()
            (a,b,c) = triangles[i]
            for x in [(a,b),(a,c),(b,c)]:
                self.genContraction(x[0], x[1])

    def contractOnce(self):
        contr = self.nextContraction()
        if contr is not None:
            self.doContraction(contr)

    def nextContraction(self):
        while True:
            if len(self.heap) == 0: return None
            contr = heapq.heappop(self.heap)
            if contr[4]: return contr

    # Compute the quadric associated with a vertex during the initialization
    # phase.
    def vertexQuadric(self, i, progress):
        progress.step()
        A = numpy.zeros((3,3))
        b = numpy.zeros((1,3))
        c = 0
        for tri_index in self.adj[i]:
            A2, b2, c2 = self.tri_quadrics[tri_index]
            A += A2/3.0
            b += b2/3.0
            c += c2/3.0
        return [A,b,c]

    def genContraction(self, i1, i2):
        if i1 > i2: i1, i2 = i2, i1
        if (i1, i2) in self.contractionsByVertices: return
        A1, b1, c1 = self.quadrics[i1]
        A2, b2, c2 = self.quadrics[i2]
        A = A1 + A2
        b = b1 + b2
        c = c1 + c2
        e1 = evalQuadric(A,b,c,self.vertices[i1])
        e2 = evalQuadric(A,b,c,self.vertices[i2])
        if e1 < e2:
            contr = [e1, self.numContr, i1, i2, True]
        else:
            contr = [e2, self.numContr, i2, i1, True]
        heapq.heappush(self.heap, contr)
        self.numContr += 1
        self.contractions[i1][contr[1]] = contr
        self.contractions[i2][contr[1]] = contr
        self.contractionsByVertices[(i1,i2)] = contr

    def doContraction(self, contr):
        err, id, i1, i2, valid = contr

        for i in range(3):
            self.quadrics[i1][i] += self.quadrics[i2][i]

        # Invalidate all the contractions that involve i2
        for id in self.contractions[i2]:
            contr = self.contractions[i2][id]
            contr[4] = False
            other_i = contr[2]
            if other_i == i2: other_i = contr[3]
            del self.contractions[other_i][contr[1]]
            if other_i < i2: tup = (other_i, i2)
            else: tup = (i2, other_i)
            del self.contractionsByVertices[tup]

        to_delete = []
        for tri_index in self.adj[i2]:
            # If triangle also contains i1, then it will become
            # degenerate. Delete it.
            if tri_index in self.adj[i1]:
                to_delete.append(tri_index)

            # Otherwise, we need to generate some new contractions.
            else:
                self.adj[i1][tri_index] = self.adj[i2][tri_index]
                self.triangles[tri_index][self.adj[i1][tri_index]] = i1
                for x in [(0,1),(0,2),(1,2)]:
                    self.genContraction(self.triangles[tri_index][x[0]],
                                        self.triangles[tri_index][x[1]])

        to_delete.sort()
        to_delete.reverse()
        for i in range(len(to_delete)):
#            print i, to_delete[i], self.triangles[to_delete[i]]
            self.swapTriangles(to_delete[i], len(self.triangles) - 1 - i)
        for i in range(len(to_delete)):
            tri_index = len(self.triangles) - 1
            # print self.triangles[tri_index]
            if tri_index not in self.adj[i1]:
                print self.triangles[tri_index]
                print i1, tri_index, i
                print self.adj[i1]
                raise Exception
            self.deleteLastTriangle()

        last = len(self.vertices) - 1
        if i2 == last:
            self.vertices.pop()
            self.adj.pop()
            self.contractions.pop()
            self.quadrics.pop()
        else:
            # Move last vertex to index i2
            for id in self.contractions[last]:
                contr = self.contractions[last][id]
                if contr[2] == last:
                    contr[2] = i2
                    other = contr[3]
                else:
                    contr[3] = i2
                    other = contr[2]
                if other < i2: tup = (other, i2)
                else: tup = (i2, other)
                self.contractionsByVertices[tup] = contr
                del self.contractionsByVertices[(other, last)]
    
            for tri_index in self.adj[last]:
                self.triangles[tri_index][self.adj[last][tri_index]] = i2

            self.vertices[i2] = self.vertices.pop()
            self.adj[i2] = self.adj.pop()
            self.contractions[i2] = self.contractions.pop()
            self.quadrics[i2] = self.quadrics.pop()


    def isValid(self):
        for i in range(len(self.vertices)):
            for j in self.adj[i]:
                if self.triangles[j][self.adj[i][j]] != i:
                    print "Adj",i,":",self.adj[i]
                    print "Triangle",j,":",self.triangles[j]
                    return False
        for i in range(len(self.triangles)):
            for j in range(3):
                if self.adj[self.triangles[i][j]][i] != j:
                    print "Triangle",i,":",self.triangles[i]
                    print "Adj",self.triangles[i][j],":",self.adj[self.triangles[i][j]]
                    return False
        return True

    def deleteLastTriangle(self):
        last = len(self.triangles) - 1
        for i in self.triangles[last]:
            del self.adj[i][last]
        self.triangles.pop()
        for attr_list in self.corner_attributes:
            attr_list.pop()

    def swapTriangles(self, i1, i2):
        if i1 == i2: return
#        print "Swapping",i1,"(",self.triangles[i1],") with",i2,"(",self.triangles[i2],")"
        for i in set(list(self.triangles[i1]) + list(self.triangles[i2])):
            if i1 in self.adj[i] and i2 in self.adj[i]:
                self.adj[i][i1], self.adj[i][i2] = self.adj[i][i2], self.adj[i][i1]
            elif i1 in self.adj[i]:
                self.adj[i][i2] = self.adj[i][i1]
                del self.adj[i][i1]
            else:
                self.adj[i][i1] = self.adj[i][i2]
                del self.adj[i][i2]
        self.triangles[i1], self.triangles[i2] = self.triangles[i2], self.triangles[i1]
        for attr_list in self.corner_attributes:
            attr_list[i1], attr_list[i2] = attr_list[i2], attr_list[i1]
