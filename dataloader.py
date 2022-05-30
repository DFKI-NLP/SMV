import json
import transformers
from warnings import warn
import thermostat

import search_algorithms as sa


class Verbalizer:
    def __init__(self, path: str, standard_samples: int = -1, model_type: str = [], len_filters: int = 5
                 , *args, **kwargs):
        """
        Verbalizer class
        :param path: Path to explained-dataset
        :param model_type: Which model is being explained? optional; will look by itself
        :param standard_samples: how many samples should be loaded if nothing specified
        """

        self.models = {
            "bert": (r"BertTokenizer", r"bert-base-uncased"),
            "albert": (r"AlbertTokenizer", r"albert-base-v2"),
            "roberta": (r"RobertaTokenizer", r"roberta-base"),
            "xlnet": (r"XLNetTokenizer", r"xlnet-base-cased"),
            "electra": (r"ElectraTokenizer", r"google/electra-small-discriminator")
        }

        self.path = path
        self.model_type = model_type
        if not model_type:
            for model in self.models.keys():
                if model in self.path:
                    self.model_type = model

        if not self.path:
            raise RuntimeError("Please add path parameter; Missing param path")
        if not self.model_type:
            raise RuntimeError("Please specify model_type; Missing param model_type")

        self.standard_samples = standard_samples
        self.modes = ["total_order", "filter search"]  # which search-algorithms to use
        self.checkpoint = 0  # where did the Verbalizer stop loading examples
        self.len_filters = len_filters
        self.label_names = None
        self.tokenizer = eval("transformers.{}.from_pretrained('{}')".format(self.models[self.model_type][0],
                                                                             self.models[self.model_type][1]))

    def read_samples(self, num_entries: int = None, recursion=0):
        """
        reads num_entries entries in self.path´s explained_dataset and converts to ordered dict
        saves checkpoint to continue from to save memory
        :param recursion: DEBUG DO NOT CHANGE, prevents too much recursion
        :param num_entries: how many samples should be loaded
        :return: ordered dict of shape {num_entries * ordered dict of shape {index:(token, value)}}
        """
        if not num_entries:
            num_entries = self.standard_samples
        _dict = {}
        if num_entries <= 0:
            self.checkpoint = 0
            with open(self.path, "r") as dataset:
                dataset = list(dataset)
            for counter in range(len(dataset)):
                _ = json.loads(dataset[counter])
                _dict[counter] = {"input_ids": self.tokenizer.convert_ids_to_tokens(_["input_ids"]),
                                  "attributions": _["attributions"],
                                  "label": _["label"],
                                  "predictions": _["predictions"]
                                  }  # why dict? dict fast
        else:
            self.checkpoint = 0
            with open(self.path, "r") as dataset:
                dataset_snippet = []
                for i in range(self.checkpoint):
                    dataset.readline()

                for i in range(num_entries):
                    dataset_snippet.append(dataset.readline())
                    self.checkpoint += 1

                for counter in range(len(dataset_snippet)):
                    _ = json.loads(dataset_snippet[counter])
                    _dict[counter] = {"input_ids": self.tokenizer.convert_ids_to_tokens(_["input_ids"]),
                                      "attributions": _["attributions"],
                                      "label": _["label"],
                                      "predictions": _["predictions"]}  # why dict? dict fast

            if not dataset_snippet:  # TODO remove recursion ?
                self.checkpoint = 0
                if recursion > 0:
                    raise RecursionError("Failed retrieving samples, after {} recursions.\n".format(recursion) +
                                         "This can be due to a unreadable dataset (of length 0) or to unknown errors.\n"
                                         + "Try checking the dataset.")
                _dict = self.read_samples(num_entries, recursion)

        # after retrieving data, we clean it from padding, cls and sep by deleting everything with 0 attribution
        cleaned_dict = {}
        for sample in _dict.keys():
            input_ids = _dict[sample]["input_ids"]
            attributions = _dict[sample]["attributions"]
            _cleaned_ids = []
            _cleaned_attributions = []

            for index in range(len(input_ids)):
                if attributions[index] != 0:
                    _cleaned_ids.append(input_ids[index])
                    _cleaned_attributions.append(attributions[index])
            cleaned_dict[sample] = {"input_ids": _cleaned_ids,
                                    "attributions": _cleaned_attributions,
                                    "label": _dict[sample]["label"],
                                    "predictions": _dict[sample]["predictions"]}
        if _:
            self.label_names = _["dataset"]["label_names"]
        else:
            self.label_names = dataset_snippet["dataset"]["label_names"]
        return cleaned_dict

    def doit(self, modes: list = None, n_samples: int = None):
        """
        verbalizes n_samples of self.path´s dataset
        :param modes: optional: which searches to do on dataset
        :param n_samples: optional: how many samples to verbalize
        :return: n_samples * (verbalized version of heatmap; by total_order, ...) as array of strings (tuple/list)
        """

        # init standard params
        if not n_samples:
            n_samples = self.standard_samples
        if not modes:
            modes = self.modes
        # end of init

        sample_array = self.read_samples(n_samples)

        orders_and_searches = {}

        """
        example of how to use search algorithm
        if "total_order" in modes:
            orders_and_searches["total_order"] = self.total_order(sample_array)
        """

        if "filter search" in modes:
            orders_and_searches["filter search"] = self.filter_search(sample_array, self.len_filters)

        if "span search" in modes:
            orders_and_searches["span search"] = self.span_search(sample_array, self.len_filters)

        return orders_and_searches, sample_array

    def __call__(self, modes: list = None, n_samples: int = None, *args, **kwargs):
        return self.doit(modes, n_samples)

    @staticmethod
    def span_search(_dict, len_filters):
        explanations = sa.span_search(_dict, len_filters)
        return explanations

    @staticmethod
    def filter_search(_dict, len_filters):
        explanations = sa.field_search(_dict, len_filters)
        return explanations

# spacy -> klassifikation von zeugs

# filterbasierte suche


# 1. wrapper
# 2. filter
# 3. spacy
