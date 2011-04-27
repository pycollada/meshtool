import heapq
from .mesh_simplification import MeshSimplification
from .progress_printer import ProgressPrinter

class AggregateSimplifier:

    def __init__(self):
        self.heap = []
        self.history = []
        self.simplifiers = {}
        self.num_vertices = 0

    def addPrimitive(self, key, vertices, triangles,
                     attributes, attribute_sources):
        self.simplifiers[key] = MeshSimplification(
            key, self.heap, self.history, vertices,
            triangles, attributes, attribute_sources)
        self.num_vertices += len(vertices)

    def contractOnce(self):
        contr = self.nextContraction()
        if contr is not None:
            self.simplifiers[contr[1]].doContraction(contr[0])

    def nextContraction(self):
        while True:
            if len(self.heap) == 0: return None
            contr = heapq.heappop(self.heap)
            if contr[0][4]: return contr

    def generatePM(self):
        pm = []
        progress = ProgressPrinter(len(self.history))
        for s in self.simplifiers.itervalues():
            s.startPM(pm)
        for i in range(len(self.history)-1, -1, -1):
            rec = self.history[i]
            progress.step()
            self.simplifiers[rec.key].processPmRecord(rec)
        return pm
