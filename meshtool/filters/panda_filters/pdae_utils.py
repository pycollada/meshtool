import sys
import struct

class PM_OP:
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2
    VERTEX_ADDITION = 3
    
def readPDAE(pm_filebuf):
    pdae_line = pm_filebuf.readline().strip()
    if pdae_line != 'PDAE':
        print >> sys.stderr, 'Progressive mesh file given does not have a valid PDAE header'
        sys.exit(1)
    
    num_refinements = int(pm_filebuf.readline().strip())

    pm_refinements = []
    for refinement_index in range(num_refinements):
        num_operations = int(pm_filebuf.readline().strip())
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
        pm_refinements.append(refinement_ops)

    return pm_refinements

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
    