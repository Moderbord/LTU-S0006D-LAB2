from os import path

import sys
import pygame as pg

import game_settings as settings
import game_tiles as tiles
import game_assets as assets
import algorithms as alg

class Game:

    def __init__(self):
        pg.init()
        pg.display.set_caption(settings.TITLE)
        self.clock = pg.time.Clock()
        self.load_data()
        self.path = None
        self.pathqueue = None
        self.pathprocess = []

    # Loads map file 
    def load_data(self):
        self.map_data = []
        with open(path.join(assets.map_folder, "Map3.txt"), "rt") as f:
            for line in f:
                self.map_data.append(line)

    def new(self):
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        # Construct tileset from map data
        tmp_walls = []
        for y, row in enumerate(self.map_data):
            for x, tile in enumerate(row):
                if (tile == "X"):
                    tiles.Wall(self, x, y)
                    tmp_walls.append((x, y))
                elif (tile == "S"):
                    tiles.Start(self, x, y)
                    self.start = (x, y)
                elif (tile == "G"):
                    tiles.Goal(self, x, y)
                    self.goal = (x, y)

        # TODO randomize start and goal positions
        
        # Constructs a sqaure graph from map data
        self.sGraph = alg.SquareGraph(len(self.map_data[0]) - 1, len(self.map_data))
        self.sGraph.walls = tmp_walls[:]
        
        # Constructs a weighed graph from map data
        self.wGraph = alg.WeightedGraph(len(self.map_data[0]) - 1, len(self.map_data))
        self.wGraph.walls = tmp_walls[:]

        # Updates screen size to loaded map
        # -1 correction for '\0' char in str
        settings.MAP_WIDTH = (len(self.map_data[0]) - 1) * settings.TILE_SIZE
        settings.HAP_HEIGHT = len(self.map_data) * settings.TILE_SIZE
        self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))

    def run(self):
        self.running = True
        while (self.running):
            self.dt = self.clock.tick(settings.FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def update(self):
        self.all_sprites.update()

        # catch inputs
        keystate = pg.key.get_pressed()
        if keystate[pg.K_ESCAPE]:
            pg.event.post(pg.event.Event(pg.QUIT))

        if keystate[pg.K_q]: # BFS
            self.pathqueue = None
            self.path = alg.BFS(self.sGraph, self.start, self.goal)

        if keystate[pg.K_a]: # Visual BFS
            self.path = alg.BFS(self.sGraph, self.start, self.goal)
            self.visual_helper()

        if keystate[pg.K_w]: # DFS
            self.pathqueue = None
            self.path = alg.DFS(self.sGraph, self.start, self.goal)

        if keystate[pg.K_s]: # Visual DFS
            self.path = alg.DFS(self.sGraph, self.start, self.goal)
            self.visual_helper()

        if keystate[pg.K_e]: # Dijkstra
            self.pathqueue = None
            self.path = alg.Dijkstra(self.wGraph, self.start, self.goal)

        if keystate[pg.K_d]: # Visual Dijkstra
            self.path = alg.Dijkstra(self.wGraph, self.start, self.goal)
            self.visual_helper()

        if keystate[pg.K_z]: # Dijkstra - only path
            self.pathqueue = None
            d_path = alg.Dijkstra(self.wGraph, self.start, self.goal)
            self.path = alg.ReconstructPath(d_path, self.start, self.goal)

        if keystate[pg.K_x]: # Visual Dijkstra - only path
            d_path = alg.Dijkstra(self.wGraph, self.start, self.goal)
            self.path = alg.ReconstructPath(d_path, self.start, self.goal)
            self.visual_helper(True)

        if keystate[pg.K_r]: # Astar
            self.pathqueue = None
            self.path = alg.Astar(self.wGraph, self.start, self.goal)

        if keystate[pg.K_f]: # Visual Astar
            self.path = alg.Astar(self.wGraph, self.start, self.goal)
            self.visual_helper()

        if keystate[pg.K_c]: # Astar - only path
            self.pathqueue = None
            d_path = alg.Astar(self.wGraph, self.start, self.goal)
            self.path = alg.ReconstructPath(d_path, self.start, self.goal)

        if keystate[pg.K_v]: # Visual Astar - only path
            d_path = alg.Astar(self.wGraph, self.start, self.goal)
            self.path = alg.ReconstructPath(d_path, self.start, self.goal)
            self.visual_helper(True)

    def visual_helper(self, use_stack=False):
        self.pathqueue = alg.Stack() if use_stack else alg.Queue()
        self.pathprocess = []
        for child, parent in self.path.items():
            if parent is None:
                continue
            chi = tuple(x * settings.TILE_SIZE + settings.TILE_SIZE / 2 for x in child)
            par = tuple(x * settings.TILE_SIZE + settings.TILE_SIZE / 2 for x in parent)
            self.pathqueue.put((chi, par))

    def draw_grid(self):
        for x in range(0, settings.MAP_WIDTH, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (x, 0), (x, settings.HAP_HEIGHT))
        for y in range(0, settings.HAP_HEIGHT, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (0, y), (settings.MAP_WIDTH, y))

    def draw_path(self):
        # Visual step-by-step representation of search
        if(self.pathqueue and not self.pathqueue.empty()):
            # Append next bit of search
            self.pathprocess.append(self.pathqueue.get())
            # Draw the current path
            for pair in self.pathprocess:
                (child, parent) = pair
                pg.draw.line(self.screen, settings.COLOR["BLACK"], child, parent)
            # Wait till next draw (ms)
            pg.time.delay(100)

        # Final visual representation
        elif(self.path):
            for child, parent in self.path.items():
                if parent == None:
                    continue
                # Correction for tilesize
                cOffset = tuple(x * settings.TILE_SIZE + settings.TILE_SIZE / 2 for x in child)
                pOffset = tuple(x * settings.TILE_SIZE + settings.TILE_SIZE / 2 for x in parent)
                pg.draw.line(self.screen, settings.COLOR["BLACK"], cOffset, pOffset)

    def draw(self):
        self.screen.fill(settings.COLOR["WHITE"])
        self.all_sprites.draw(self.screen)
        self.draw_grid()
        self.draw_path()
        pg.display.flip()

    def events(self):
        # catch events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

game = Game()
game.new()
game.run()
