# importing all modules
from pyneat.activations import clamped, tanh
import pygame as pg  # simulation module

from pyneat.population import Population
from pyneat.options import Options
from pyneat.population import Population 

from collections import deque  # deque is just a list with maximum size and removes old items automatically
from random import randint  # just to set the pipe's y position

pg.init()
pg.font.init()

draw_lines = False

screen_width, screen_height = 400, 600
screen = pg.display.set_mode((screen_width, screen_height))

bg_image = pg.transform.scale(pg.image.load("Images/background.gif"), (screen_width, screen_height - 100))

# defining the font and some colors
font = pg.font.Font('freesansbold.ttf', 32)
WHITE = 255, 255, 255
RED = 255, 0, 0
GREEN = 0, 255, 0

# defining the fps or the speed of the game (can be increased or decreased using up and down arrow keys)
fps = 2**10
fps_clock = pg.time.Clock()

# the bird class
class Bird:
    def __init__(self, nn):
        self.x = screen_width // 4
        self.y = screen_height // 2

        self.nn = nn

        # loading the image of the bird and defining a rect for collision detection
        self.dim = screen_height // 12

        self.image = pg.transform.scale(pg.image.load("Images/bird.gif"), (self.dim, self.dim))
        self.rect = None

        # how much to change y position of the bird when jumping and falling
        self.jump_power = 50
        self.fall_power = 15

    def display(self):
        # displaying the bird and making it fall
        self.fall()
        self.rect = screen.blit(self.image, (self.x, self.y))

    def jump(self):
        # making the bird jump if it doesn't go out of the screen (upwards)
        if self.y > 0:
            self.y -= self.jump_power

    def fall(self):
        # making the bird fall
        self.y += self.fall_power

# class for the pipe(s)
class Pipe:
    def __init__(self):
        # loading the top and bottom pipes' images
        self.image_top = pg.transform.scale(pg.image.load("Images/pipet.gif"), (screen_width // 4, screen_height))
        self.image_bottom = pg.transform.scale(pg.image.load("Images/pipeb.gif"), (screen_width // 4, screen_height))

        # the rect of both (for collision detection)
        self.top_rect = None
        self.bottom_rect = None

        # defining the gap between the top and bottom pipes, the x position and the random y position
        self.gap = 150
        self.x = screen_width
        self.y = randint(0, screen_height - 100 - self.gap)

    def display(self):
        # displaying the pipes
        self.top_rect = screen.blit(self.image_top, (self.x, self.y - self.image_top.get_height()))
        self.bottom_rect = screen.blit(self.image_bottom, (self.x, self.y + self.gap))

    def move(self):
        # moving the pipes to the left
        self.x -= 10

    def check_collision(self, bird_rect):
        # collision detection with the bird (notice the rect of the bird)
        return self.top_rect.colliderect(bird_rect) or self.bottom_rect.colliderect(bird_rect)

    def reached_bird(self):
        # just getting info if the pipe has reached the bird
        return self.x + self.image_top.get_width() == screen_width // 4

# class for the ground (I did not intend to make a new whole new class for the ground but added it anyway)
class Ground:
    def __init__(self):
        # as usual the loading the images but not the rect (reason given later)
        self.image = pg.transform.scale(pg.image.load("Images/ground.gif"), (screen_width, screen_height // 6))

        # the x and y position of the ground and it is just static
        self.x = 0
        self.y = screen_height - self.image.get_height()

    def display(self):
        # displaying the ground
        self.rect = screen.blit(self.image, (self.x, self.y))

    def collide_with_ground(self, bird_obj):
        # not using rect as it is an extensive process and collision can simply be checked by using y position
        # if bird's y is greater than ground's y then it has collided
        return bird_obj.y > self.y

# the main controller or the environment
class GameEnv:
    def __init__(self):
        # the list for the birds
        self.birds = []

        # list for pipes and DEQUE ALERT!!!
        self.pipes = deque(maxlen=2)  # notice the maxlen
        self.pipes.append(Pipe())  # adding the first pipe

        # initialising some extra details
        self.score = 0
        self.generation = 0  # seems akward but okay?!
        self.alive_birds = len(self.birds)

        # and finally the ground
        self.ground = Ground()

    def add_bird(self, nn):
        self.birds.append(Bird(nn))

    def display_all(self):
        # nothing fancy just displaying everything-

        # 1. all birds
        for bird in self.birds:
            bird.display()

        # 2. all pipes
        for pipe in self.pipes:
            pipe.display()

        # 3. the ground
        self.ground.display()

    def check_removal_birds(self):
        # main site of checking collisions

        # define a list, add all collided birds to the list and remove them from the birds list at the end
        # because removing it on the go gives an error
        birds_to_remove = []

        # check for all birds
        for bird in self.birds:
            # finally, THE BIRD RECT
            bird_rect = bird.rect

            # and with pipes
            for pipe in self.pipes:

                # check collision with the ground
                if self.ground.collide_with_ground(bird):
                    birds_to_remove.append(bird)
                    break

                if pipe.check_collision(bird_rect):
                    birds_to_remove.append(bird)
                    break

        # now just remove the collided birds from the main birds list
        for rem in birds_to_remove:
            rem.nn.fitness -= 1
            self.birds.remove(rem)

        self.alive_birds = len(self.birds)

    def move_pipes(self):
        # just like the previous problem - gives an error if pipes are added on the go
        pipes_to_append = []

        # iterating over all pipes
        for pipe in self.pipes:
            # move the pipe
            pipe.move()

            # if the pipe reaches the birds then just update the score
            if pipe.reached_bird():
                self.score += 1
                pipes_to_append.append(Pipe())  # and add a new pipe, the old pipe that is beyond the screen is automatically deleted (because of deque)

        # Tip: to add list to another list use extend instead append
        self.pipes.extend(pipes_to_append)  # just add the new pipes

    def get_info(self, bird):
        # this is just the pipe ahead of the bird
        pipe = self.pipes[-1]

        # if draw_lines is True then just show "what the bird sees"
        width = pipe.image_top.get_width()
        if draw_lines:
            pg.draw.line(screen, RED, (bird.x, bird.y), (pipe.x + width, pipe.y), 3)
            pg.draw.line(screen, GREEN, (bird.x, bird.y), (pipe.x + width, pipe.y + pipe.gap), 3)

        # and finally return the information
        return (
            bird.y,  # bird's y position
            pipe.y,  # top pipe's y position
            pipe.y + pipe.gap,  # bottom pipe's y position
            pipe.x  # and pipe's x position
        )

    def move_birds(self):
        # moving the birds according to their brains or the NEAT net
        for bird in self.birds:
            # For more distance keep adding the fitness (i.e. more distance -> more fitness)
            bird.nn.fitness += 1

            # just think according to the information given by the above method
            output = bird.nn.predict(self.get_info(bird))

            if output[0] > 0:
                bird.jump()

    def all_dead(self):
        # just a helper method to know if all birds are dead
        return not len(self.birds) > 0

    def reset(self):
        # just the initial setting is called when reset is called
        self.birds = []
        self.score = 0

        self.pipes = deque(maxlen=2)
        self.pipes.append(Pipe())

        self.ground = Ground()

        # as this will only be called when one generation finishes, so increase the generation number
        self.generation += 1

# defining the main game env
##################
env = GameEnv() ##
##################

def eval_genomes(nns):
    global fps

    for nn in nns:
        env.add_bird(nn)

    # the main loop
    while True:
        # draw the background
        screen.blit(bg_image, (0, 0))

        # the environment functions
        env.move_pipes()
        env.move_birds()
        env.display_all()
        env.check_removal_birds()

        # if all birds are dead just reset the environment and break the loop, the neat-python module will handle it automatically
        if env.all_dead():
            env.reset()
            break

        # just displaying the extra information
        screen.blit(font.render(f"Generation: {env.generation}", True, WHITE), (10, screen_height - 100))
        screen.blit(font.render(f"Score: {env.score}", True, WHITE), (10, screen_height - 70))
        screen.blit(font.render(f"Birds Alive: {env.alive_birds}", True, WHITE), (10, screen_height - 40))

        # the event loop
        for event in pg.event.get():
            # if you click the close button, then close the window and stop the program
            if event.type == pg.QUIT:
                pg.quit()
                quit()

            # this just to control the fps
            if event.type == pg.KEYDOWN:
                # up key to increase the fps
                if event.key == pg.K_UP:
                    fps *= 2
                # down key to decrease the fps
                elif event.key == pg.K_DOWN:
                    fps /= 2

        # update the screen and respect the fps (just do everything according to the fps)
        pg.display.update()
        fps_clock.tick(fps)

def main():
    Options.set_options(4, 1, 200, 100000000, activation_func=tanh, target_species=10, excess_coeff=2, disjoint_coeff=2, weight_coeff=2, weight_mutate_prob=0.5)

    p = Population()
    p.evaluate(eval_genomes, 1000000)

# and this is just calling the main function
if __name__ == '__main__':
    main()

# stopping pygame
pg.quit()
