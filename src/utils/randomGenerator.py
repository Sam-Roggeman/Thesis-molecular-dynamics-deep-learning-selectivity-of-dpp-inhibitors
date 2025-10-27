import numpy as np

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class RandomGenerator(metaclass=Singleton):
    def __init__(self, seed=None):
        if seed:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()
            self.seed = None

    def generate_random_unit_vector(self):
        random_vector = self.rng.normal(size=3)
        random_vector /= np.linalg.norm(random_vector)  # Normalize the random vector