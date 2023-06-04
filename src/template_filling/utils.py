import random


class TemplateElement:
    """ Sample from a dict of options with probabilities. """
    def __init__(self, sample_options):
        assert round(sum(sample_options.values()), 2) == 1.0, "Probabilities do not sum to 1."
        self.filler_keys = list(sample_options.keys())
        self.filler_values = list(sample_options.values())

    def sample(self, k=1):
        return random.choices(self.filler_keys, self.filler_values, k=k)[0]

    def __call__(self, *args, **kwargs):
        return self.sample()
