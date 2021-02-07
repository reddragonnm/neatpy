import random
from pygame.draw import circle, line, rect
from pygame.color import THECOLORS as colors
from pygame import Rect

_species_colors = {}

def draw_brain_pygame(screen, brain, x=50, y=50, dim=300, circle_size=15, line_width=4):
    info = brain.get_draw_info()

    for conn in info['connections']['enabled']:
        line(
            screen,
            colors['dodgerblue'] if conn['weight'] > 0 else colors['coral'],
            (int(dim * conn['from'][1] + x), int(dim * conn['from'][0] + y)),
            (int(dim * conn['to'][1] + x), int(dim * conn['to'][0] + y)),
            line_width
        )

    for inp in info['nodes']['input']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)

    for inp in info['nodes']['bias']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)

    for inp in info['nodes']['hidden']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)

    for inp in info['nodes']['output']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)

def draw_species_bar_pygame(screen, population, x=0, y=0, width=400, height=100):
    s = sum(sp.spawns_required for sp in population.species) + 1

    counter = 0

    for sp in population.species:
        color = _species_colors.get(sp.id)
        if color is None:
            color = tuple(random.randrange(255) for _ in range(3))
            _species_colors[sp.id] = color

        w = int((sp.spawns_required / s) * width)
        r = Rect(counter + x, y, w, height)

        rect(screen, color, r)
        rect(screen, colors['black'], r, 2)

        counter += w

        