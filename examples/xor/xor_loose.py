from neatpy.options import Options
from neatpy.population import Population

Options.set_options(2, 1, 150)
p = Population()

xor_inp = [(0,0), (0,1), (1,0), (1,1)]
xor_out = [0, 1, 1, 0]

max_fitness = 3.9 # maximum fitness

# while the maximum fitness hasn't been reached
while p.best.fitness < max_fitness:
    for nn in p.pool:
        nn.fitness = 4
    
        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2

    p.epoch() # create a new pool
    print(p) # __str__ method gives the statistics