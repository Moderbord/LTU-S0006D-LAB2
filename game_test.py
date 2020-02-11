from os import path

import sys
import pygame as pg

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import random

import game_settings as settings
import game_tiles as tiles
import game_assets as assets
import algorithms as alg
import map_data



class Game:

    def __init__(self):
        pg.init()
        pg.display.set_caption(settings.TITLE)
        self.clock = pg.time.Clock()
        self.all_sprites = pg.sprite.Group()
        self.path = None
        self.pathqueue = None
        self.pathprocess = []
        self.map = map_data.TileMap()


    # Loads map file 
    def load_map(self, map_name):

        self.map.load_map_template(map_name)

        # Updates screen size to loaded map
        # -1 correction for '\0' char in str
        settings.MAP_WIDTH = self.map.map_width * settings.TILE_SIZE
        settings.HAP_HEIGHT = self.map.map_height * settings.TILE_SIZE
        self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))

    # # Loads map file 
    # def load_map_old(self, map_name):
    #     self.walls = pg.sprite.Group()
        
    #     # parse file
    #     m_width = 0
    #     m_height = 0
    #     tmp_walls = []
    #     with open(path.join(assets.map_folder, map_name), "rt") as f:
    #         # Construct tileset from map data
    #         for y, row in enumerate(f):
    #             m_height += 1
    #             m_width = 0
    #             for x, tile in enumerate(row):
    #                 m_width += 1
    #                 if (tile == "X"):
    #                     tiles.Wall(self, x, y)
    #                     tmp_walls.append((x, y))
    #                 elif (tile == "S"):
    #                     tiles.Start(self, x, y)
    #                     self.start = (x, y)
    #                 elif (tile == "G"):
    #                     tiles.Goal(self, x, y)
    #                     self.goal = (x, y)

    #     # TODO randomize start and goal positions
        
    #     # Constructs a sqaure graph from map data
    #     self.square_graph = alg.SquareGraph(m_width, m_height)
    #     self.square_graph.walls = tmp_walls[:]
        
    #     # Constructs a weighed graph from map data
    #     self.weighed_graph = alg.WeightedGraph(m_width, m_height)
    #     self.weighed_graph.walls = tmp_walls[:]

    #     # Updates screen size to loaded map
    #     # -1 correction for '\0' char in str
    #     settings.MAP_WIDTH = m_width * settings.TILE_SIZE
    #     settings.HAP_HEIGHT = m_height * settings.TILE_SIZE
    #     self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))

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
            self.path = alg.BFS(alg.SquareGraph(self.map), self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_a]: # Visual BFS
            self.path = alg.BFS(alg.SquareGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.visual_helper()

        if keystate[pg.K_w]: # DFS
            self.pathqueue = None
            self.path = alg.DFS(alg.SquareGraph(self.map), self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_s]: # Visual DFS
            self.path = alg.DFS(alg.SquareGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.visual_helper()

        if keystate[pg.K_e]: # Dijkstra
            self.pathqueue = None
            self.path = alg.Dijkstra(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_d]: # Visual Dijkstra
            self.path = alg.Dijkstra(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.visual_helper()

        if keystate[pg.K_z]: # Dijkstra - only path
            self.pathqueue = None
            d_path = alg.Dijkstra(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_x]: # Visual Dijkstra - only path
            d_path = alg.Dijkstra(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.map.custom_start, self.map.custom_goal)
            self.visual_helper(True)

        if keystate[pg.K_r]: # Astar
            self.pathqueue = None
            self.path, cost = alg.Astar(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_f]: # Visual Astar
            self.path, cost = alg.Astar(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.visual_helper()

        if keystate[pg.K_c]: # Astar - only path
            self.pathqueue = None
            d_path, cost = alg.Astar(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.map.custom_start, self.map.custom_goal)

        if keystate[pg.K_v]: # Visual Astar - only path
            d_path, cost = alg.Astar(alg.WeightedGraph(self.map), self.map.custom_start, self.map.custom_goal)
            self.path = alg.ReconstructPath(d_path, self.map.custom_start, self.map.custom_goal)
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
game.load_map("Map3.txt")
game.run()
