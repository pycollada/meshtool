from heapq import heappush, heappop
from networkx import NetworkXError
import networkx as nx
import __builtin__

def astar_path(G, source, target, heuristic=None, weight='weight', exclude=None, subset=None):
    """Return a list of nodes in a shortest path between source and target 
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.
    
    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path 

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    exclude: set, optional
       An optional set of nodes that will be excluded from
       the traversal
       
    subset: set, optional
       An optional set of nodes that the path has to be in

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> print(nx.astar_path(G,0,4))
    [0, 1, 2, 3, 4]
    >>> G=nx.grid_graph(dim=[3,3])  # nodes are two-tuples (x,y)
    >>> def dist(a, b):
    ...    (x1, y1) = a
    ...    (x2, y2) = b
    ...    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G,(0,0),(2,2),dist))
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]


    See Also
    --------
    shortest_path, dijkstra_path

    """
    if G.is_multigraph():
        raise NetworkXError("astar_path() not implemented for Multi(Di)Graphs")
    
    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u,v):
            return 0
    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add each node's hash to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guarenteed unique for all nodes in the graph.
    queue = [(0, hash(source), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}
    
    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = heappop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            if neighbor in explored or \
                    (exclude and neighbor != target and neighbor in exclude) or \
                    (subset and neighbor not in subset):
                continue
            ncost = dist + w.get(weight,1)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost < ncost, a longer path to neighbor remains
                # enqueued. Removing it would need to filter the whole
                # queue, it's better just to leave it there and ignore
                # it when we visit the node a second time.
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            heappush(queue, (ncost + h, hash(neighbor), neighbor, 
                             ncost, curnode))

    raise nx.exception.NetworkXError("Node %s not reachable from %s"%(source,target))


def dfs_interior_nodes(G, starting, boundary, subset):
    """Produce nodes on the interior of a boundary of nodes
    
    Parameters
    ----------
    G : NetworkX graph

    starting : set
       Starting nodes to find interior nodes from
       
    boundary : set
       Boundary of nodes not to go outside of
       
    subset : set
       List of nodes to consider. Any nodes not in this set
       will not be traversed.
    
    """
    nodes=list(starting)
    visited=set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [iter(G[start])]
        while stack:
            children = stack[-1]
            try:
                child = next(children)
                if child not in visited and child not in boundary and child in subset:
                    yield child
                    visited.add(child)
                    stack.append(iter(G[child]))
            except StopIteration:
                stack.pop()
                
def super_cycle(G):
    """Yields the nodes of the longest cycle available in G"""
    
    cycles = nx.cycle_basis(G)
    if len(cycles) < 1:
        return
    cycle_sets = [set(c) for c in cycles]
    visited_cycles = set()
    
    def visit_cycle(curcycle, startnode):
        thiscycle = cycles[curcycle]
        reordered_cycle = thiscycle[thiscycle.index(startnode):] + thiscycle[:thiscycle.index(startnode)]
        visited_cycles.add(curcycle)
        for node in reordered_cycle:
            for i, othercycle in enumerate(cycle_sets):
                if i not in visited_cycles and node in othercycle:
                    for othernode in visit_cycle(i, node):
                        yield othernode
            yield node
    
    return visit_cycle(0, cycles[0][0])
