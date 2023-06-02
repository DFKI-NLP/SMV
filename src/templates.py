import random
from typing import List


class Synonym:
    """ Sample from a list of options without placeholders. """
    def __init__(self):
        self.OPTIONS = []

    def sample(self):
        return random.choice(self.OPTIONS)


class Template:
    """ Build a string out of options and rules. """



class UnitsPlusVerb(Template):
    # TODO: Capitalization of unit_name at the start of a sentence
    OPTIONS = [
        f"{u1} is"
        f"{unit_name} {u1} is"
    ]
    # TODO: Plural forms


class MultipleUnits(Template):
    OPTIONS = [
        f'The two {unit_name}s {u1} and {u2}',
        f'Both {unit_name}s {u1} and {u2}',
        f'{u1} and {u2} are both {important_name}',
        f'The three most {important_name} {unit_name}s are {u1}, {u2}, and {u3}',
        f'The top three most {important_name} {unit_name}s are {u1}, {u2}, and {u3}',
        f'{unit_name}s such as {u1} and {u2}',
    ]


class SingleFeatureName(Synonym):
    def __init__(self):
        super().__init__()
        self.OPTIONS = [
            "feature",
            "word",
            "token",
        ]


class SpanName(Synonym):
    def __init__(self):
        super().__init__()
        self.OPTIONS = [
            "span",
            "phrase",
        ]


class ImportantName(Synonym):
    def __init__(self):
        super().__init__()
        self.OPTIONS = [
            "important",
            "salient",
            "influential",
            "impactful"
        ]

class PredictionName(Synonym):
    def __init__(self):
        super().__init__()
        self.OPTIONS = [
            'classification',
            'decision',
            'model prediction',
            'model\'s prediction',
            'model\'s judgment',
            'model behavior',
            'model\'s behavior',
            'prediction of the classifier',
            'predicted label',
            'model\'s predicted label',
            'outcome',
        ]

class Conjunctions(Template):
    TEMPLATES = [
        f'{v1}, while {units_plus_verb} also salient',
        f'{v1}, whereas {units_plus_verb} also salient',
        f'{v1} with the word {u1} also being salient'
    ]


class ImportantAdditions(Template):
    OPT_TO_THE_MODEL = [
        "",
        "to the model",
    ]

    TEMPLATES = [
        f'{leading_sentence} {OPT_TO_THE_MODEL} for the {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} for this {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} in making this {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} in predicting this {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} in choosing this {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} in producing this {prediction_name}',
        f'{leading_sentence} {OPT_TO_THE_MODEL} in shaping this {prediction_name}',
        f'{leading_sentence} with respect to the {prediction_name}',
        f'{leading_sentence} in this text',
    ]


class ImportantVariations(Template):
    TEMPLATES = [
        f'{units_plus_verb} focused on the most for this {prediction_name}',
        f'{units_plus_verb} used by the model to make its {prediction_name}',
        f'{units_plus_verb} caused the model to predict this {prediction_name}',
        f'{units_plus_verb} indicate the {prediction_name}',
        f'{units_plus_verb} shaped the {prediction_name}'
        f'{units_plus_verb} shaped the {prediction_name} the most'
    ]


class Polarity(Template):
    @staticmethod
    def value_polarity(x, polarity=" "):
        assert polarity in [" ", " not "]
        return f"{x}{polarity}salient"

    @staticmethod
    def value_superlative(x, superlative="most"):
        assert superlative in ["most", "least"]
        return f"{x} {superlative} salient"

    @staticmethod
    def value_comparison(x, y, comparative="more"):
        assert comparative in ["more", "less"]
        # TODO: allow any except the word with the lowest saliency
        # select any word that has a lower saliency than 1
        return f"{x} {comparative} salient than {y}"
