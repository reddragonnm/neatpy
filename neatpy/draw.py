from pygame.draw import circle, line
from pygame.color import THECOLORS as colors

def draw_brain_pygame(screen, brain, x=50, y=50, dim=300, circle_size=15, line_width=4, node_border_thickness=1):
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
        circle(screen, colors['black'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size, node_border_thickness)

    for inp in info['nodes']['bias']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)
        circle(screen, colors['black'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size, node_border_thickness)

    for inp in info['nodes']['hidden']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)
        circle(screen, colors['black'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size, node_border_thickness)

    for inp in info['nodes']['output']:
        circle(screen, colors['white'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size)
        circle(screen, colors['black'], (int(dim * inp[1] + x), int(dim * inp[0] + y)), circle_size, node_border_thickness)