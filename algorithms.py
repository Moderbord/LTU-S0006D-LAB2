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

class SquareGraph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        #results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)] # 4x movement
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # 8x movement

        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results
    
def BFS(graph, start, goal):
    front = Queue()
    front.put(start)
    path = {}
    path[start] = None

    while (not front.empty()):
        current = front.get()

        if (current == goal):
            break

        for next in graph.neighbors(current):
            if (next not in path):
                front.put(next)
                path[next] = current

    return path

