import panda3d.core as p3d
from meshtool.filters.panda_filters import pdae_utils as pdae

def add_refinements(geomPath, refinements):
    geom = geomPath.node().modifyGeom(0)
    vertdata = geom.modifyVertexData()
    prim = geom.modifyPrimitive(0)
    indexdata = prim.modifyVertices()
    
    indexrewriter = p3d.GeomVertexRewriter(indexdata)
    indexrewriter.setColumn(0)
    nextTriangleIndex = indexdata.getNumRows()
    
    vertwriter = p3d.GeomVertexWriter(vertdata, 'vertex')
    numverts = vertdata.getNumRows()
    vertwriter.setRow(numverts)
    normalwriter = p3d.GeomVertexWriter(vertdata, 'normal')
    normalwriter.setRow(numverts)
    uvwriter = p3d.GeomVertexWriter(vertdata, 'texcoord')
    uvwriter.setRow(numverts)
    
    for refinement in refinements:
        for vals in refinement:
            op = vals[0]
            if op == pdae.PM_OP.TRIANGLE_ADDITION:
                indexrewriter.setRow(nextTriangleIndex)
                nextTriangleIndex += 3
                indexrewriter.addData1i(vals[1])
                indexrewriter.addData1i(vals[2])
                indexrewriter.addData1i(vals[3])
            
            elif op == pdae.PM_OP.INDEX_UPDATE:
                indexrewriter.setRow(vals[1])
                indexrewriter.setData1i(vals[2])
                
            elif op == pdae.PM_OP.VERTEX_ADDITION:
                numverts += 1
                vertwriter.addData3f(vals[1], vals[2], vals[3])
                normalwriter.addData3f(vals[4], vals[5], vals[6])
                uvwriter.addData2f(vals[7], vals[8])

    indexdata.setNumRows(nextTriangleIndex)
