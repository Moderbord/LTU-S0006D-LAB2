import pygame as pg
import sys
from os import path

import game_settings as settings
import game_tiles as tiles
import game_assets as assets


class Game:

    def __init__(self):
        pg.init()
        pg.display.set_caption(settings.TITLE)
        self.screen = pg.display.set_mode((settings.MAP_WIDTH, settings.HAP_HEIGHT))
        self.clock = pg.time.Clock()
        self.load_data()


    def load_data(self):
        self.map_data = []
        with open(path.join(assets.map_folder, "Map1.txt"), "rt") as f:
            for line in f:
                self.map_data.append(line)

    def new(self):
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for row, cells in enumerate(self.map_data):
            for col, cell in enumerate(cells):
                if (cell == "X"):
                    tiles.Wall(self, col, row)
                elif (cell == "S"):
                    tiles.Start(self, col, row)
                elif (cell == "G"):
                    tiles.End(self, col, row)

        # -1 correction for '\0 char
        self.screen = pg.display.set_mode(((len(self.map_data[0]) - 1) * settings.TILE_SIZE, len(self.map_data) * settings.TILE_SIZE))

    def run(self):
        self.running = True
        while (self.running):
            self.dt = self.clock.tick(settings.FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def update(self):
        self.all_sprites.update()

    def draw_grid(self):
        for x in range(0, settings.MAP_WIDTH, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (x, 0), (x, settings.HAP_HEIGHT))
        for y in range(0, settings.HAP_HEIGHT, settings.TILE_SIZE):
            pg.draw.line(self.screen, settings.COLOR["LIGHTGRAY"], (0, y), (settings.MAP_WIDTH, y))

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
