import pygame as pg
import sys
from os import path

import game_settings as settings
import game_tiles as tiles
import game_assets as assets
import algorithms as alg


class Game:

    def __init__(self):
        pg.init()
        pg.display.set_caption(settings.TITLE)
        #self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))
        self.clock = pg.time.Clock()
        self.load_data()
        self.path = None


    def load_data(self):
        self.map_data = []
        with open(path.join(assets.map_folder, "Map3.txt"), "rt") as f:
            for line in f:
                self.map_data.append(line)

    def new(self):
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        tmpWalls = []
        for y, row in enumerate(self.map_data):
            for x, tile in enumerate(row):
                if (tile == "X"):
                    tiles.Wall(self, x, y)
                    tmpWalls.append((x, y))
                elif (tile == "S"):
                    tiles.Start(self, x, y)
                    self.start = (x, y)
                elif (tile == "G"):
                    tiles.Goal(self, x, y)
                    self.goal = (x, y)

        self.sGraph = alg.SquareGraph(len(self.map_data[0]) - 1, len(self.map_data))
        self.sGraph.walls = tmpWalls[:]
        

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
        if keystate[pg.K_b]:
            self.path = alg.BFS(self.sGraph, self.start, self.goal)


    def draw_grid(self):
        for x in range(0, settings.MAP_WIDTH, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (x, 0), (x, settings.HAP_HEIGHT))
        for y in range(0, settings.HAP_HEIGHT, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (0, y), (settings.MAP_WIDTH, y))

        if(self.path):
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
