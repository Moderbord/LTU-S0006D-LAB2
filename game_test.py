import sys
import pygame as pg

import game_settings as settings
import game_tiles as tiles
import game_assets as assets
import algorithms as alg
import map_data
import nnetw



class Game:

    def __init__(self):
        pg.init()
        pg.display.set_caption(settings.TITLE)
        self.clock = pg.time.Clock()
        self.sprite_group_all = pg.sprite.Group()
        self.path = None
        self.pathqueue = None
        self.pathprocess = []
        self.tilemap = map_data.TileMap()


    # Loads tilemap file 
    def load_map(self, map_name):

        self.tilemap.load_map_template(map_name)

        # Updates screen size to loaded tilemap
        # -1 correction for '\0' char in str
        settings.MAP_WIDTH = self.tilemap.map_width * settings.TILE_SIZE
        settings.HAP_HEIGHT = self.tilemap.map_height * settings.TILE_SIZE
        self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))

        if len(self.tilemap.walls) > 0:
            self.sprite_group_walls = pg.sprite.Group()
            for wall in self.tilemap.walls:
                (x, y) = wall
                tiles.Wall(self, x, y)

        if self.tilemap.custom_start:
            (x, y) = self.tilemap.custom_start
            tiles.Start(self, x, y)

        if self.tilemap.custom_goal:
            (x, y) = self.tilemap.custom_goal
            tiles.Goal(self, x, y)        

    def run(self):
        self.running = True
        while (self.running):
            self.dt = self.clock.tick(settings.FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def update(self):
        self.sprite_group_all.update()

        # catch inputs
        keystate = pg.key.get_pressed()
        if keystate[pg.K_ESCAPE]:
            pg.event.post(pg.event.Event(pg.QUIT))

        if keystate[pg.K_p]:
            self.sprite_group_all.empty()
            self.tilemap.randomize_start_goal(self)

        if keystate[pg.K_q]: # BFS
            self.pathqueue = None
            self.path = alg.BFS(alg.SquareGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_a]: # Visual BFS
            self.path = alg.BFS(alg.SquareGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper()

        if keystate[pg.K_w]: # DFS
            self.pathqueue = None
            self.path = alg.DFS(alg.SquareGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_s]: # Visual DFS
            self.path = alg.DFS(alg.SquareGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper()

        if keystate[pg.K_e]: # Dijkstra
            self.pathqueue = None
            self.path = alg.Dijkstra(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_d]: # Visual Dijkstra
            self.path = alg.Dijkstra(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper()

        if keystate[pg.K_z]: # Dijkstra - only path
            self.pathqueue = None
            d_path = alg.Dijkstra(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_x]: # Visual Dijkstra - only path
            d_path = alg.Dijkstra(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper(True)

        if keystate[pg.K_r]: # Astar
            self.pathqueue = None
            self.path, cost = alg.Astar(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_f]: # Visual Astar
            self.path, cost = alg.Astar(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper()

        if keystate[pg.K_c]: # Astar - only path
            self.pathqueue = None
            d_path, cost = alg.Astar(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.tilemap.custom_start, self.tilemap.custom_goal)

        if keystate[pg.K_v]: # Visual Astar - only path
            d_path, cost = alg.Astar(alg.WeightedGraph(self.tilemap), self.tilemap.custom_start, self.tilemap.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.tilemap.custom_start, self.tilemap.custom_goal)
            self.visual_helper(True)

    def visual_helper(self, use_stack=False):
        self.pathqueue = alg.Stack() if use_stack else alg.Queue()
        self.pathprocess = []
        for child, parent in self.path.items():
            # first node has no parent
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
        self.sprite_group_all.draw(self.screen)
        self.sprite_group_walls.draw(self.screen)
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
game.load_map("Map3.txt")
nn = nnetw.NeuralNetwork(game)
nn.train()