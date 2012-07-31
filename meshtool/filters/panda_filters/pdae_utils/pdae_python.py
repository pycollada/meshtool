import struct
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

class PM_OP:
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2
    VERTEX_ADDITION = 3

def readPDAEHeader(pm_filebuf):
    pdae_line = pm_filebuf.readline().strip()
    num_refinements = int(pm_filebuf.readline().strip())
    
    return pdae_line, num_refinements

def readPDAEnumops(pm_filebuf):
    return int(pm_filebuf.readline().strip())

def readPDAErefinement(pm_filebuf, num_operations):
    refinement_ops = []
    for operation_index in range(num_operations):
        vals = pm_filebuf.readline().strip().split()
        op = vals.pop(0)
        if op == 't':
            v1, v2, v3 = map(int, vals)
            refinement_ops.append((PM_OP.TRIANGLE_ADDITION, v1, v2, v3))
        elif op == 'u':
            tindex, vindex = map(int, vals)
            refinement_ops.append((PM_OP.INDEX_UPDATE, tindex, vindex))
        elif op == 'v':
            vx, vy, vz, nx, ny, nz, s, t = map(float, vals)
            refinement_ops.append((PM_OP.VERTEX_ADDITION, vx, vy, vz, nx, ny, nz, s, t))
        else:
            print op
            assert(False)
            
    return refinement_ops

def readPDAE(pm_filebuf):
    pdae_line, num_refinements = readPDAEHeader(pm_filebuf)
    if pdae_line != 'PDAE':
        return None

    pm_refinements = []
    for refinement_index in range(num_refinements):
        num_operations = readPDAEnumops(pm_filebuf)
        refinement_ops = readPDAErefinement(pm_filebuf, num_operations)
        pm_refinements.append(refinement_ops)

    return pm_refinements

def readPDAEPartial(data, refinements_read, num_refinements):
    fakebuf = StringIO(data)
    
    lines_left = data.count('\n')
    
    if num_refinements is None:
        pdae_line, num_refinements = readPDAEHeader(fakebuf)
        lines_left -= 2
    
    pm_refinements = []
    while lines_left > 5 and refinements_read < num_refinements:
        num_operations = readPDAEnumops(fakebuf)
        lines_left -= 1
        
        if lines_left < num_operations:
            break

        refinement_ops = readPDAErefinement(fakebuf, num_operations)
        lines_left -= len(refinement_ops)
        refinements_read += 1
        pm_refinements.append(refinement_ops)
        num_operations = None
    
    data_left = data[fakebuf.tell():]
    if num_operations is not None:
        data_left = "%d\n" % num_operations + data_left
    
    return (refinements_read, num_refinements, pm_refinements, data_left)

def writeBPDAE(pm_refinements, bpdae_outbuf):
    #6-byte magic number header - the string BPDAE followed by a null character
    bpdae_outbuf.write(struct.pack('!6s', 'BPDAE'))
    
    #the number of refinements in the file
    bpdae_outbuf.write(struct.pack('!L', len(pm_refinements)))
    
    for refinement in pm_refinements:
        #number of objects in this refinement
        bpdae_outbuf.write(struct.pack('!H', len(refinement)))
        
        for operation in refinement:
            vals = list(operation)
            op = vals.pop(0)
            
            if op == PM_OP.TRIANGLE_ADDITION:
                v1, v2, v3 = vals
                bpdae_outbuf.write(struct.pack('!cIII', 't', v1, v2, v3))
            elif op == PM_OP.INDEX_UPDATE:
                tindex, vindex = vals
                bpdae_outbuf.write(struct.pack('!cII', 'u', tindex, vindex))
            elif op == PM_OP.VERTEX_ADDITION:
                vx, vy, vz, nx, ny, nz, s, t = vals
                bpdae_outbuf.write(struct.pack('!cffffffff', 'v', vx, vy, vz, nx, ny, nz, s, t))
    
    bpdae_outbuf.close()
