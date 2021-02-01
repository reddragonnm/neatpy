from neatpy.options import Options
from neatpy.population import Population

Options.set_options(2, 1, 150, 3.9)
p = Population()

xor_inp = [(0,0), (0,1), (1,0), (1,1)]
xor_out = [0, 1, 1, 0]

max_fitness = 3.9

while p.best.fitness < max_fitness:
    for nn in p.pool:
        nn.fitness = 4
    
        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2

    p.epoch()
    print(p)

best = p.best

# def evaluate(nns):
#     for nn in nns:
        


# p = Population()
# best, solved = p.evaluate(evaluate, 400)

# evaluate([best])
# print(best.fitness)
