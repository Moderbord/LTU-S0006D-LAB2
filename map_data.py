from os import path

import game_assets as assets

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
            # set map width
            self.map_width = len(f.readline())
            # Construct tileset from map data
            for y, row in enumerate(f):
                # count map height
                self.map_height += 1
                # new inner array
                inner_array = []
                for x, tile in enumerate(row):
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