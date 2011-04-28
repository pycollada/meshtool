import collada
import numpy

def geomReplacePrimitive(geom, prim_index, vertices, triangles, attributes,
                         attribute_sources):
    base_id = geom.id+"-"+str(prim_index)
    il = collada.source.InputList()
    prim = geom.primitives[prim_index]
    new_id = base_id+"-vertex"
    while new_id in geom.sourceById: new_id += "-x"
    vertexsource = collada.source.FloatSource(
        new_id, numpy.array(vertices), ('X', 'Y', 'Z'))
    geom.sourceById[new_id] = vertexsource
    offset = 0
    il.addInput(offset, "VERTEX", "#"+new_id)
    offset += 1
    last = len(attributes) - 1
    if prim.normal is not None:
        new_id = base_id+"-normal"
        while new_id in geom.sourceById: new_id += "-x"
        normalsource = collada.source.FloatSource(
            new_id, numpy.array(attribute_sources[last]),
            ('X', 'Y', 'Z'))
        geom.sourceById[new_id] = normalsource
        il.addInput(offset, "NORMAL", "#"+new_id)
        offset += 1
    if prim.texcoordset is not None:
        texcoord_indexset = []
        for i in range(last):
            new_id = base_id+"-texcoord"+str(i)
            while new_id in geom.sourceById: new_id += "-x"
            texcoordsource = collada.source.FloatSource(
                new_id, numpy.array(attribute_sources[i]),
                ('S', 'T'))
            geom.sourceById[new_id] = texcoordsource
            il.addInput(offset, "TEXCOORD", "#"+new_id)
            offset += 1
            texcoord_indexset.append(numpy.array(attributes[i]))
    indices = numpy.array(triangles)
    indices.shape = (-1,3,1)
    if prim.normal is not None:
        normal_index = numpy.array(attributes[last])
        normal_index.shape = (-1,3,1)
        indices = numpy.append(indices, normal_index, 2)
    if prim.texcoordset is not None:
        for texcoord_index in texcoord_indexset:
            texcoord_index.shape = (-1,3,1)
            indices = numpy.append(indices, texcoord_index, 2)
    new_prim = geom.createTriangleSet(indices.flatten(), il, prim.material)
    geom.primitives[prim_index] = new_prim

    # Clean up unused sources
    referenced_sources = {}
    for prim in geom.primitives:
        for semantic in prim.sources:
            for input in prim.sources[semantic]:
                referenced_sources[input[2][1:]] = True
    unreferenced_sources = []
    for id in geom.sourceById:
        if id not in referenced_sources:
            unreferenced_sources.append(id)
    for id in unreferenced_sources:
        del geom.sourceById[id]
