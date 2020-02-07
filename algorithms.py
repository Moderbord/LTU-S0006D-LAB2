import collections

class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()

class Stack:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.pop()

class SquareGraph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, neighbor):
        (x, y) = neighbor
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, neighbor):
        return neighbor not in self.walls

    def cutting_corner(self, current, neighbor):
        (x, y) = neighbor
        # movement from current to neighbor
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        # optimization (movement is not diagonal and check can be skipped)
        if (dx * dy == 0):
            return True
        # possible blocking walls
        posibleWalls = [(x-dx, y), (x, y-dy)]
        # If any of the neighbors is a wall return false
        return posibleWalls[0] not in self.walls and posibleWalls[1] not in self.walls

    def neighbors(self, current):
        (x, y) = current
        #currentNeighbors = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)] # 4x movement
        currentNeighbors = [(x+1, y), (x, y-1), (x-1, y), (x, y+1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # 8x movement

        currentNeighbors = filter(self.in_bounds, currentNeighbors)
        currentNeighbors = filter(self.passable, currentNeighbors)
        currentNeighbors = filter(lambda neighbor: self.cutting_corner(current, neighbor), currentNeighbors)
        return currentNeighbors
    
def BFS(graph, start, goal):
    front = Queue()
    front.put(start)
    path = {}
    path[start] = None

    while (not front.empty()):
        # Pops next in queue
        current = front.get()

        # Goal found
        if (current == goal):
            break
        
        # Iterate through current neighbors
        for node in graph.neighbors(current):
            # If neighbor isn't in current path
            if (node not in path):
                front.put(node)
                path[node] = current

    return path

def DFS(graph, start, goal):
    path = {}
    path[start] = None
    found = False

    def inner(current):

        for cell in graph.neighbors(current):

            if(current == goal):
                return goal

            elif cell not in path:
                path[cell] = current
                if(inner(cell) == goal):
                    return goal

    inner(start)
    return path