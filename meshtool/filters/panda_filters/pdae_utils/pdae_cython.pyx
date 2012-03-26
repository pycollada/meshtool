from libc.stdlib cimport strtod, strtol
from libc.stdio cimport fgets

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

cdef enum PM_OP:
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2
    VERTEX_ADDITION = 3

cdef readPDAEHeader(pm_filebuf):
    pdae_line = pm_filebuf.readline().strip()
    line = pm_filebuf.readline()
    cdef char* lineptr = line
    num_refinements = strtol(lineptr, NULL, 10)
    return pdae_line, num_refinements

cdef int readPDAEnumops(pm_filebuf):
    line = pm_filebuf.readline()
    cdef char* lineptr = line
    return strtol(lineptr, NULL, 10)

cdef list readPDAErefinement(pm_filebuf, int num_operations):
    cdef list refinement_ops
    cdef int operation_index
    cdef char op
    cdef list vals
    cdef long int t1, t2, t3
    cdef long int u1, u2, u3
    cdef float f1, f2, f3, f4, f5, f6, f7, f8

    cdef char* line
    cdef char* ptr
    cdef char** ptrptr = &ptr
    
    refinement_ops = []
    for operation_index in range(num_operations):
        str = pm_filebuf.readline()
        line = str
        op = line[0]
        if op == b't':
            t1 = strtol(line+1, ptrptr, 10)
            t2 = strtol(ptr, ptrptr, 10)
            t3 = strtol(ptr, ptrptr, 10)
            refinement_ops.append((TRIANGLE_ADDITION, t1, t2, t3))
        elif op == b'u':
            u1 = strtol(line+1, ptrptr, 10)
            u2 = strtol(ptr, ptrptr, 10)
            refinement_ops.append((INDEX_UPDATE, u1, u2))
        elif op == b'v':
            f1 = strtod(line+1, ptrptr)
            f2 = strtod(ptr, ptrptr)
            f3 = strtod(ptr, ptrptr)
            f4 = strtod(ptr, ptrptr)
            f5 = strtod(ptr, ptrptr)
            f6 = strtod(ptr, ptrptr)
            f7 = strtod(ptr, ptrptr)
            f8 = strtod(ptr, ptrptr)
            refinement_ops.append((VERTEX_ADDITION, f1, f2, f3, f4, f5, f6, f7, f8))
        else:
            print op
            assert(False)
            
    return refinement_ops

def readPDAE(pm_filebuf):
    cdef int num_refinements
    cdef int refinement_index
    cdef int num_operations
    cdef list pm_refinements
    cdef list refinement_ops
    
    pdae_line, num_refinements = readPDAEHeader(pm_filebuf)
    if pdae_line != 'PDAE':
        return None

    pm_refinements = []
    for refinement_index in range(num_refinements):
        num_operations = readPDAEnumops(pm_filebuf)
        refinement_ops = readPDAErefinement(pm_filebuf, num_operations)
        pm_refinements.append(refinement_ops)
    
    return pm_refinements

def readPDAEPartial(data, int refinements_read, num_refinements):
    fakebuf = StringIO(data)
    
    cdef int lines_left = data.count('\n')
    
    if num_refinements is None:
        pdae_line, num_refinements = readPDAEHeader(fakebuf)
        lines_left -= 2
    
    cdef list pm_refinements = []
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
