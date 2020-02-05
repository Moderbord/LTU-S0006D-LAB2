

# Map size
TILE_SIZE = 16
HAP_HEIGHT = 512
MAP_WIDTH = 512

TITLE = "ALGO"
FPS = 30

# Colours
COLOR = { 
    "BLACK" : (0, 0, 0),
    "WHITE" : (255, 255, 255),
    "LIGHTGRAY" : (100, 100, 100),
    "RED" : (255, 0, 0),
    "GREEN" : (0, 255, 25)
    }

def GridWidth():
    return MAP_WIDTH / TILE_SIZE

def GridHeight():
    return HAP_HEIGHT / MAP_WIDTH
