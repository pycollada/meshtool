import math

class CouldNotPack:
    pass

class TreeNode:
    def __init__(self, left, right, rect, key):
        self.left = left
        self.right = right
        self.rect = rect
        self.key = key

    def __iter__(self):
        if self.left is not None:
            for k in self.left:
                yield k
        if self.key is not None:
            yield (self.key, self.rect)
        if self.right is not None:
            for k in self.right:
                yield k

def insert(node, rect):
    if node.left is not None:
        newNode = insert(node.left, rect)
        if newNode is not None:
            return newNode
        return insert(node.right, rect)
    if node.key is not None: return None
    lx, ly, lw, lh = node.rect
    k, w, h = rect
    dw = lw - w
    dh = lh - h
    if dw < 0 or dh < 0: return None
    if dw == 0 and dh == 0:
        node.key = k
        return node
    if dw > dh:
        node.left = TreeNode(None, None, (lx, ly, w, lh), None)
        node.right = TreeNode(None, None, (lx + w, ly, lw - w, lh), None)
    else:
        node.left = TreeNode(None, None, (lx, ly, lw, h), None)
        node.right = TreeNode(None, None, (lx, ly + h, lw, lh - h), None)
    return insert(node.left, rect)

# Sort by longer side then shorter side, descending
def rectcmp(rect1, rect2):
    (k1, w1, h1) = rect1
    (k2, w2, h2) = rect2
    if w1 < h1: w1, h1 = h1, w1
    if w2 < h2: w2, h2 = h2, w2
    if w1 > w2: return -1
    if w2 > w1: return 1
    if h1 > h2: return -1
    if h2 > h1: return 1
    return 0

class RectPack:
    def __init__(self, maxwidth=None, maxheight=None):
        self.maxwidth = maxwidth
        self.maxheight = maxheight
        self.rectangles = {}

    def addRectangle(self, key, width, height):
        self.rectangles[key] = (width, height)

    def pack(self):
        rects = [(key, self.rectangles[key][0], self.rectangles[key][1])
                 for key in self.rectangles]
        rects.sort(rectcmp)
        
        #initial smallest pack could be two 1x1 rects, although not likely
        width = 2
        height = 2
        
        done = False
        while not done:
            locations = TreeNode(None, None, (0,0,width,height), None)
            self.rejects = []
            
            try:
                for rect in rects:
                    if insert(locations, rect) is None:
                        if self.maxwidth and width >= self.maxwidth or \
                            self.maxheight and height >= self.maxheight:
                            self.rejects.append(rect[0])
                        else:
                            raise CouldNotPack()
                done = True
            except CouldNotPack:
                width *= 2
                height *= 2
                
        self.placements = {}
        for key, rect in locations:
            self.placements[key] = rect
        self.width = width
        self.height = height
        
        if len(self.rejects) > 0:
            return False
        else:
            return True

    def getPlacement(self, key):
        return self.placements[key]
