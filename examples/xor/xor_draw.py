from neatpy.options import Options
from neatpy.population import Population
from neatpy.draw import draw_brain_pygame
from neatpy.save import save_brain

import pygame as pg
from pygame.color import THECOLORS as colors

pg.init()
screen = pg.display.set_mode((400, 400))

Options.set_options(2, 1, 150, 3.9)

p = Population()

xor_inp = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_out = [0, 1, 1, 0]

max_fitness = 3.9

while p.best.fitness < max_fitness:
    screen.fill(colors['lightgreen'])

    for nn in p.pool:
        nn.fitness = 4

        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2

    draw_brain_pygame(screen, p.best, dim=250, x=75,
                      y=40, circle_size=20, line_width=7)

    p.epoch()
    print(p)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            quit()

    pg.display.update()

save_brain(p.best, 'brain_test')
pg.image.save(screen, 'file.png')
