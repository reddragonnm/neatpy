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
```python
from neatpy.options import Options # import options class for configuration
from neatpy.population import Population # import population class

# XOR inputs and outputs
xor_inp = [(0,0), (0,1), (1,0), (1,1)]
xor_out = [0, 1, 1, 0]

# evaluate function
def evaluate(brains):
    for nn in brains:
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
## Notes
- For every environment an `evaluate` function needs to be created that takes every generation's population
- To customize the algorithm even further, optional arguments in `Options.set_options` can be tweaked
- When the evaluation ends the method returns the best brain and whether the environment was solved (maximum fitness reached) or not

## XOR Loose example
Imitating the evaluate method in Population() we can write the above code as:
```python
from neatpy.options import Options
from neatpy.population import Population

Options.set_options(2, 1, 150, 3.9)
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

best = p.best # best brain 
```
### Enjoy 🥳
