import numpy
from numpy import array
import copy
import heapq
import math
from .progress_printer import ProgressPrinter
import collada
import pprint

def array_mult(arr1, arr2):
    return arr1[:,0]*arr2[:,0] + arr1[:,1]*arr2[:,1] + arr2[:,2]*arr1[:,2]
def array_dot(arr1, arr2):
    return numpy.sqrt( array_mult(arr1, arr2) )

def quadricsForTriangles(tris):
    normal = numpy.cross( tris[::,1] - tris[::,0], tris[::,2] - tris[::,0] )
    collada.util.normalize_v3(normal)
    
    s1 = tris[::,1] - tris[::,0]
    s1 = array_dot(s1, s1)
    s2 = tris[::,2] - tris[::,0]
    s2 = array_dot(s2, s2)
    s3 = tris[::,2] - tris[::,1]
    s3 = array_dot(s3, s3)
    
    sp = (s1 + s2 + s3) / 2.0
    area = sp*(sp-s1)*(sp-s2)*(sp-s3)
    area_zeros = numpy.zeros(area.shape, area.dtype)
    area = numpy.where(area < 0, area_zeros, numpy.sqrt(area))
    area_zeros = None
    
    d = -array_mult(normal, tris[:,0])

    b2 = normal * (area*d)[:,numpy.newaxis]
    c2 = area*d*d
    
    A2 = numpy.dstack((normal[:,0][:,numpy.newaxis] * normal,
                       normal[:,1][:,numpy.newaxis] * normal,
                       normal[:,2][:,numpy.newaxis] * normal))
    A2 = area[:,numpy.newaxis,numpy.newaxis] * A2
    
    return (A2, b2, c2, area, normal)

def evalQuadric(A, b, c, pt):
    return numpy.dot(pt,numpy.inner(A,pt)) + 2*numpy.dot(b,pt) + c

class ContractionRecord:
    # Contains:
    # key - the key of the simplifier that created this record
    # source - index of vertex that is deleted
    # target - index of vertex that is contracted to
    # deleted_triangles - list of deleted triangles (by index)
    # deleted_triangles_opp_v - list of opposite vertices from contracted edge
    #                           in deleted triangles (by index)
    # deleted_triangles_perm - for each deleted triangle, a permutation of
    #                          (0,1,2), indicating which of the three corners
    #                          is the source, the target, and the opposite corner.
    # changed_triangles - list of changed triangles (by index)
    def __init__(self, key, source, target, deleted_triangles,
                 deleted_triangles_opp_v, deleted_triangles_perm,
                 changed_triangles):
        self.key = key
        self.source = source
        self.target = target
        self.deleted_triangles = deleted_triangles
        self.deleted_triangles_opp_v = deleted_triangles_opp_v
        self.deleted_triangles_perm = deleted_triangles_perm
        self.changed_triangles = changed_triangles

class MeshSimplification:

    def __init__(self, key, heap, history, vertices, triangles, attributes=[],
                 attribute_sources=[]):
        print "Copying..."
        self.key = key
        self.original_vertices = vertices
        self.original_triangles = triangles
        self.original_attributes = attributes
        self.original_attribute_sources = attribute_sources
        self.vertices = [vertex for vertex in vertices]
        self.vertex_indices = range(len(vertices))
        self.history = history
        self.triangles = [copy.copy(triangle) for triangle in triangles]
        self.triangle_indices = range(len(triangles))
        self.attributes = [[copy.copy(triangle) for triangle in attr_list]
                           for attr_list in attributes]
        self.attribute_sources = [[attr for attr in attr_list]
                                  for attr_list in attribute_sources]
        self.attribute_indices = [range(len(attr_list))
                                  for attr_list in attribute_sources]
        self.adj = [{} for v in vertices]
        self.attribute_adj = [[{} for attr in attr_list]
                              for attr_list in attribute_sources]
        self.adj_edge = {}
        print "Building simplex..."
        progress = ProgressPrinter(len(triangles))
        for i in range(len(triangles)):
            progress.step()
            (a,b,c) = triangles[i]
            for x in [(a,b,c),(a,c,b),(b,c,a)]:
                x1, x2, x3 = x
                if x1 > x2: x1, x2 = x2, x1
                if (x1, x2) not in self.adj_edge:
                    self.adj_edge[(x1, x2)] = []
                self.adj_edge[(x1, x2)].append(x3)
            self.adj[a][i] = 0
            self.adj[b][i] = 1
            self.adj[c][i] = 2
            for j in range(len(self.attribute_adj)):
                (a,b,c) = attributes[j][i]
                self.attribute_adj[j][a][i] = True
                self.attribute_adj[j][b][i] = True
                self.attribute_adj[j][c][i] = True

        print "Generating triangle quadrics..."
        progress = ProgressPrinter(len(triangles))
        
        self.tri_quadrics = quadricsForTriangles(vertices[triangles])
        self.avg_area = numpy.mean(self.tri_quadrics[3])
        
        print "Computing vertex quadrics..."
        progress = ProgressPrinter(len(vertices))
        self.quadrics = [self.vertexQuadric(i, progress) for i in range(len(vertices))]
        del self.tri_quadrics # Not needed anymore
        del self.adj_edge     # Not needed anymore
        self.heap = heap
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

    def startPM(self, pm):
        self.pm = pm
        self.new_v_indices = [-1 for v in self.original_vertices]
        self.num_v = len(self.vertex_indices)
        for i in range(self.num_v):
            self.new_v_indices[self.vertex_indices[i]] = i
        self.new_t_indices = [-1 for t in self.original_triangles]
        self.num_t = len(self.triangle_indices)
        for i in range(self.num_t):
            self.new_t_indices[self.triangle_indices[i]] = i
        self.new_a_indices = [[-1 for a in attr_list]
                              for attr_list in self.original_attribute_sources]
        self.num_a = [len(attr_list) for attr_list in self.attribute_indices]
        for attr_id in range(len(self.attribute_indices)):
            for i in range(len(self.attribute_indices[attr_id])):
                self.new_a_indices[attr_id][
                    self.attribute_indices[attr_id][i]] = i

    def processPmRecord(self, rec):
        coords = self.original_vertices[rec.source]
        split_index = self.new_v_indices[rec.target]
        if split_index == -1: raise Exception
        changed_triangles = [self.new_t_indices[i]
                             for i in rec.changed_triangles]
        new_triangles_opp_v = [self.new_v_indices[i]
                               for i in rec.deleted_triangles_opp_v]
        new_triangles_flip = []
        new_triangles_attr = [[] for x in self.original_attributes]
        for j in range(len(rec.deleted_triangles_perm)):
            i1, i2, i3 = rec.deleted_triangles_perm[j]
            flip = (i1,i2,i3) in [(0,2,1),(2,1,0),(1,0,2)]
            new_triangles_flip.append(flip)
            tri_index = rec.deleted_triangles[j]
            for k in range(len(new_triangles_attr)):
                attr_orig = self.original_attributes[k][tri_index]
                attr_cur = []
                if flip: ord = (i2, i1, i3)
                else: ord = (i1, i2, i3)
                for ind in ord:
                    a = attr_orig[ind]
                    cur_i = self.new_a_indices[k][a]
                    if cur_i == -1:
                        attr_cur.append(
                            self.original_attribute_sources[k][a])
                        self.new_a_indices[k][a] = self.num_a[k]
                        self.num_a[k] += 1
                    else:
                        attr_cur.append(cur_i)
                new_triangles_attr[k].append(attr_cur)
        self.pm.append((self.key, split_index, coords, changed_triangles,
                        new_triangles_opp_v, new_triangles_flip,
                   new_triangles_attr))
        self.new_v_indices[rec.source] = self.num_v
        self.num_v += 1
        for tri_i in rec.deleted_triangles:
            self.new_t_indices[tri_i] = self.num_t
            self.num_t += 1

    # Compute the quadric associated with a vertex during the initialization
    # phase.
    def vertexQuadric(self, i, progress):
        progress.step()
        A = numpy.zeros((3,3))
        b = numpy.zeros((1,3))
        c = 0
        for tri_index in self.adj[i]:
            A2, b2, c2, area, normal = (self.tri_quadrics[ii][tri_index] for ii in range(5))
            for i2 in self.triangles[tri_index]:
                if i2 == i: continue
                x1, x2 = min(i2, i), max(i2, i)
                if len(self.adj_edge[(x1, x2)]) == 1:
                    v = self.vertices[i] - self.vertices[i2]
                    normal2 = numpy.cross(v, normal)
                    normal2 = normal2 / numpy.linalg.norm(normal2)
                    d = -numpy.dot(normal, self.vertices[i])
                    A3 = 3*self.avg_area*numpy.outer(normal2, normal2)
                    b3 = 3*self.avg_area*d*normal2
                    c3 = 3*self.avg_area*d*d
                    A += A3
                    b += b3
                    c += c3
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
        heapq.heappush(self.heap, (contr, self.key))
        self.numContr += 1
        self.contractions[i1][contr[1]] = contr
        self.contractions[i2][contr[1]] = contr
        self.contractionsByVertices[(i1,i2)] = contr

    def doContraction(self, contr):
        err, id, i1, i2, valid = contr

        record = ContractionRecord(self.key, self.vertex_indices[i2],
                                   self.vertex_indices[i1], [], [], [], [])

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
                record.deleted_triangles.append(self.triangle_indices[tri_index])
                for i in (0,1,2):
                    if self.triangles[tri_index][i] == i1:
                        target = i
                    elif self.triangles[tri_index][i] == i2:
                        source = i
                    else:
                        opposite = i
                record.deleted_triangles_opp_v.append(self.vertex_indices[
                        self.triangles[tri_index][opposite]])
                record.deleted_triangles_perm.append((source, target, opposite))

            # Otherwise, we need to generate some new contractions.
            else:
                record.changed_triangles.append(self.triangle_indices[tri_index])
                self.adj[i1][tri_index] = self.adj[i2][tri_index]
                self.triangles[tri_index][self.adj[i1][tri_index]] = i1
                for x in [(0,1),(0,2),(1,2)]:
                    self.genContraction(self.triangles[tri_index][x[0]],
                                        self.triangles[tri_index][x[1]])

        self.history.append(record)

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
            self.vertex_indices.pop()
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
            self.vertex_indices[i2] = self.vertex_indices.pop()
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
        self.triangle_indices.pop()
        for i in range(len(self.attributes)):
            x = self.attributes[i].pop()
            to_delete = []
            for attr_i in x:
                if last in self.attribute_adj[i][attr_i]:
                    del self.attribute_adj[i][attr_i][last]
                    if not self.attribute_adj[i][attr_i]:
                        to_delete.append(attr_i)
            
            to_delete.sort()
            to_delete.reverse()
            for j in range(len(to_delete)):
                self.swapAttributes(i, to_delete[j], len(self.attribute_sources[i]) - 1 - j)
            for j in range(len(to_delete)):
                self.deleteLastAttribute(i)

                    
    def swapTriangles(self, i1, i2):
        if i1 == i2: return
#        print "Swapping",i1,"(",self.triangles[i1],") with",i2,"(",self.triangles[i2],")"
        for i in set(list(self.triangles[i1]) + list(self.triangles[i2])):
            if i1 in self.adj[i] and i2 in self.adj[i]:
                self.adj[i][i1], self.adj[i][i2] = \
                    self.adj[i][i2], self.adj[i][i1]
            elif i1 in self.adj[i]:
                self.adj[i][i2] = self.adj[i][i1]
                del self.adj[i][i1]
            else:
                self.adj[i][i1] = self.adj[i][i2]
                del self.adj[i][i2]
        for attr_id in range(len(self.attributes)):
            for i in set(list(self.attributes[attr_id][i1]) +
                         list(self.attributes[attr_id][i2])):
                if i1 in self.attribute_adj[attr_id][i] and \
                        i2 in self.attribute_adj[attr_id][i]:
                    self.attribute_adj[attr_id][i][i1], \
                        self.attribute_adj[attr_id][i][i2] = \
                        self.attribute_adj[attr_id][i][i2], \
                        self.attribute_adj[attr_id][i][i1]
                elif i1 in self.attribute_adj[attr_id][i]:
                    self.attribute_adj[attr_id][i][i2] = \
                        self.attribute_adj[attr_id][i][i1]
                    del self.attribute_adj[attr_id][i][i1]
                else:
                    self.attribute_adj[attr_id][i][i1] = \
                        self.attribute_adj[attr_id][i][i2]
                    del self.attribute_adj[attr_id][i][i2]
        self.triangles[i1], self.triangles[i2] = \
            self.triangles[i2], self.triangles[i1]
        self.triangle_indices[i1], self.triangle_indices[i2] = \
            self.triangle_indices[i2], self.triangle_indices[i1]
        for attr_list in self.attributes:
            attr_list[i1], attr_list[i2] = attr_list[i2], attr_list[i1]

    def swapAttributes(self, attr_id, i1, i2):
        if i1 == i2: return
        for i in set(list(self.attribute_adj[attr_id][i1]) +
                     list(self.attribute_adj[attr_id][i2])):
            for j in (0,1,2):
                if self.attributes[attr_id][i][j] == i1:
                    self.attributes[attr_id][i][j] = i2
                elif self.attributes[attr_id][i][j] == i2:
                    self.attributes[attr_id][i][j] = i1
        self.attribute_adj[attr_id][i1], self.attribute_adj[attr_id][i2] = \
            self.attribute_adj[attr_id][i2], self.attribute_adj[attr_id][i1]
        self.attribute_sources[attr_id][i1], \
            self.attribute_sources[attr_id][i2] = \
            self.attribute_sources[attr_id][i2], \
            self.attribute_sources[attr_id][i1]
        self.attribute_indices[attr_id][i1], \
            self.attribute_indices[attr_id][i2] = \
            self.attribute_indices[attr_id][i2], \
            self.attribute_indices[attr_id][i1]

    def deleteLastAttribute(self, attr_id):
        self.attribute_adj[attr_id].pop()
        self.attribute_sources[attr_id].pop()
        self.attribute_indices[attr_id].pop()
