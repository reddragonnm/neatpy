import neat_min as n1
import neat_new.neat as n2

import random

# 933352207515909476: conn
# 5363492979423432831: weight
# 4997622651678856348: node

rnd1 = random.Random(4997622651678856348)
rnd2 = random.Random(4997622651678856348)

n1.Options.set_options(3, 2, 1)
n2.Options.set_options(3, 2, 1)

b1 = n1.Brain(1)
b2 = n2.Brain()


for _ in range(100):
    b1.mutate(rnd1)
    b2.mutate(rnd2)
    print('-'*100)
