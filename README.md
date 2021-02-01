# neatpy
neatpy is a library that implements the NEAT algorithm designed by Kenneth O. Stanley which is documented in this [paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). This method evolves neural network topologies along with weights on the foundation of a genetic algorithm.

## Why neatpy?
Some reasons to use neatpy:
- Easy to use
- Code is easier to understand
- Can be translated to other languages without much difficulty
- Little to no knowledge of configuration is required to get started
  
## Installation
```
pip install neatpy
```

## Basic XOR example
```
from neatpy.options import Options # import options class for configuration
from neatpy.population import Population # import population class

# XOR inputs and outputs
xor_inp = [(0,0), (0,1), (1,0), (1,1)]
xor_out = [0, 1, 1, 0]

# evaluate function
def evaluate(nns):
    for nn in nns:
        nn.fitness = 4
    
        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2

# number of inputs - number of outputs - population size - maximum fitness needed
Options.set_options(2, 1, 150, 3.9)

p = Population()
best, solved = p.evaluate(evaluate, 400) # evaluating population for 400 generations or till maximum fitness is reached

# testing the best neural network
evaluate([best])
print(best.fitness)
```
