from pygame import sprite
from pygame import Surface

import game_settings as settings
import game_assets as assets


class Start(sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        sprite.Sprite.__init__(self, self.groups) # Add self to group
        self.game = game
        #self.image = assets.LoadSprite("unicorn.jpg")
        self.image = Surface((settings.TILE_SIZE, settings.TILE_SIZE))
        self.image.fill(settings.COLOR["GREEN"])
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        
        # can be moved into update 
        self.rect.x = x * settings.TILE_SIZE
        self.rect.y = y * settings.TILE_SIZE

class End(sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        sprite.Sprite.__init__(self, self.groups) # Add self to group
        self.game = game
        #self.image = assets.LoadSprite("unicorn.jpg")
        self.image = Surface((settings.TILE_SIZE, settings.TILE_SIZE))
        self.image.fill(settings.COLOR["RED"])
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        
        # can be moved into update 
        self.rect.x = x * settings.TILE_SIZE
        self.rect.y = y * settings.TILE_SIZE

class Wall(sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.walls
        sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = Surface((settings.TILE_SIZE, settings.TILE_SIZE))
        self.image.fill(settings.COLOR["BLACK"])
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * settings.TILE_SIZE
        self.rect.y = y * settings.TILE_SIZE