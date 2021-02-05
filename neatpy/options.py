from .activations import sigmoid

class Options:

    @staticmethod
    def set_options(
        num_inputs,
        num_outputs,
        population_size,

        fitness_threshold=float('inf'),
        max_nodes=float('inf'),

        activation_func=sigmoid,
        aggregation_func=sum,

        excess_coeff=1,
        disjoint_coeff=1,
        weight_coeff=0.5,

        add_node_prob=0.02,
        add_conn_prob=0.05,

        weight_mutate_prob=0.1,
        new_weight_prob=0.1,
        weight_init_range=1,
        weight_mutate_power=0.5,

        feature_selection=False,

        compatibility_threshold=3,
        dynamic_compatibility_threshold=True,

        target_species=20,
        dropoff_age=15,
        survival_rate=0.2,
        species_elitism=True,

        crossover_rate=1,
        tries_tournament_selection=3,

        young_age_threshhold=10,
        young_age_fitness_bonus=1.3,
        old_age_threshold=50,
        old_age_fitness_penalty=0.7
    ):
        """Hyperparameters of the NEAT algorithm

        Args:
            num_inputs (int): Number of inputs
            num_outputs (int): Number of output
            population_size (int): Size of the population
            fitness_threshold (float): Maximum fitness required before terminating
            max_nodes (int, optional): Maximum number of nodes in the Brain. Defaults to float('inf').
            activation_func (Callable[[float], float], optional): The activation function applied to every node. Defaults to sigmoid.
            aggregation_func (Callable[[List[float]], float], optional): The aggregation of inputs in each node. Defaults to sum.
            excess_coeff (float, optional): Coefficient of the excess genes. Defaults to 1.
            disjoint_coeff (float, optional): Coefficient of the disjoint genes. Defaults to 1.
            weight_coeff (float, optional): Coefficient of the weight average. Defaults to 0.5.
            add_node_prob (float, optional): Probability to add a node. Defaults to 0.02.
            add_conn_prob (float, optional): Probability to add a connection. Defaults to 0.05.
            weight_mutate_prob (float, optional): Probability to mutate a weight. Defaults to 0.1.
            new_weight_prob (float, optional): Probability that a new weight is chosen if the weight is being mutated. Defaults to 0.1.
            weight_init_range (int, optional): [description]. Defaults to 1.
            weight_mutate_power (float, optional): Mutation power of a weight. Defaults to 0.5.
            feature_selection (bool, optional): If enabled the brain is instantiated with only one connection. Defaults to False.
            compatibility_threshold (float, optional): Threshold of the compatibility between 2 brains. Defaults to 3.
            dynamic_compatibility_threshold (bool, optional): Can the compatibility_threshold be changed so that number of species matches target_species. Defaults to True.
            target_species (int, optional): The target species required. Valid only if dynamic_compatibility_threshold is enabled. Defaults to 20.
            dropoff_age (int, optional): The number of generations after which a species is killed if it hasn't improved. Defaults to 15.
            survival_rate (float, optional): The top percentage of members per species that are allowed to reproduce. Defaults to 0.2.
            species_elitism (bool, optional): If True then the best brain of each species is automatically included in the next generation. Defaults to True.
            crossover_rate (float, optional): The percentage by which crossover should occur instead of just mutation. Defaults to 1.
            tries_tournament_selection (int, optional): Number of tries in tournament selection. Defaults to 3.
            young_age_threshhold (int, optional): A species is considered young if age is smaller than this number. Defaults to 10.
            young_age_fitness_bonus (float, optional): The bonus in fitness given if a species is considered young. Defaults to 1.3.
            old_age_threshold (int, optional): A species is considered old if age is greater than this number. Defaults to 50.
            old_age_fitness_penalty (float, optional): The penalty given to a species if it is considered old. Defaults to 0.7.
        """
        Options.num_inputs = num_inputs
        Options.num_outputs = num_outputs
        Options.population_size = population_size

        Options.fitness_threshold = fitness_threshold
        Options.max_nodes = max_nodes

        Options.activation_func = activation_func
        Options.aggregation_func = aggregation_func

        Options.excess_coeff = excess_coeff
        Options.disjoint_coeff = disjoint_coeff
        Options.weight_coeff = weight_coeff

        Options.add_node_prob = add_node_prob     
        Options.add_conn_prob = add_conn_prob   

        Options.weight_mutate_prob = weight_mutate_prob
        Options.new_weight_prob = new_weight_prob
        Options.weight_init_range = weight_init_range
        Options.weight_mutate_power = weight_mutate_power

        Options.feature_selection = feature_selection

        Options.compatibility_threshold = compatibility_threshold
        Options.dynamic_compatibility_threshold = dynamic_compatibility_threshold

        Options.target_species = target_species
        Options.dropoff_age = dropoff_age
        Options.survival_rate = survival_rate
        Options.species_elitism = species_elitism

        Options.crossover_rate = crossover_rate
        Options.tries_tournament_selection = tries_tournament_selection

        Options.young_age_threshhold = young_age_threshhold
        Options.young_age_fitness_bonus = young_age_fitness_bonus
        Options.old_age_threshold = old_age_threshold
        Options.old_age_fitness_penalty = old_age_fitness_penalty