from os import path
import random

import game_assets as assets
import game_tiles as tiles

class TileMap:

    def __init__(self):
        self.tile_data = []
        self.walls = []
        self.map_width = 0
        self.map_height = 0
        self.custom_start = None
        self.custom_goal = None

    def load_map_template(self, template_name):
        # Open map template
        with open(path.join(assets.map_folder, template_name), "rt") as f:

            # Construct tileset from map data
            for y, row in enumerate(f):
                # count map height
                self.map_height += 1
                self.map_width = 0
                # new inner array
                inner_array = []
                for x, tile in enumerate(row):
                    # count map width (lazy mf)
                    self.map_width += 1
                    if tile == "X":
                        inner_array.append("1")
                        self.walls.append((x, y))
                    else:
                        inner_array.append("0")

                    if (tile == "S"):
                        self.custom_start = (x, y)
                    elif (tile == "G"):
                        self.custom_goal = (x, y)

                self.tile_data.append(inner_array)

    def randomize_start_goal(self, game=None):
        s_ok = False
        g_ok = False

        while not s_ok:
            x1 = int((random.random())*self.map_width)
            y1 = int((random.random())*self.map_height)
            if (x1, y1) not in self.walls:
                s_ok = True

        while not g_ok:
            x2 = int((random.random())*self.map_width)
            y2 = int((random.random())*self.map_height)
            if (x2, y2) not in self.walls and (x1, y1) != (x2, y2):
                g_ok = True

        self.custom_start = (x1, y1)
        self.custom_goal = (x2, y2)

        if game:
            tiles.Start(game, x1, y1)
            tiles.Goal(game, x2, y2)

    def random_astar(self):
        pass
    