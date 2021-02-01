from pyneat import Population, Options

xor_inp = [(0,0), (0,1), (1,0), (1,1)]
xor_out = [0, 1, 1, 0]

def evaluate(nns):
    for nn in nns:
        nn.fitness = 4
    
        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2

Options.set_options(2, 1, 150, 3.9)

p = Population()
best, solved = p.evaluate(evaluate, 400)

# print(p.gen, solved)

evaluate([best])
print(best.fitness)
