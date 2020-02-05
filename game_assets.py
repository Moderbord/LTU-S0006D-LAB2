from os import path
from pygame import image

# Assets folders
# __file__ = current file dir
game_folder = path.dirname(__file__)
sprite_folder = path.join(game_folder, "assets\sprites")
map_folder = path.join(game_folder, "assets\maps")

def LoadSprite(asset_name):
    return image.load(os.path.join(sprite_folder, asset_name)).convert()