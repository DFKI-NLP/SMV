import json
import multiprocessing
import time

import transformers
from tqdm import tqdm
from typing import List, Union
import psutil
from multiprocessing import Process, Pool

import src.search_methods.tools as t
import src.processing.shared_methods as sm
import src.processing.shared_value_searches as svs


class Verbalizer:
    def __init__(self, source: Union[str, List], standard_samples: int = -1, model_type: str = [], len_filters: int = 5,
                 config=None, dev=False, multiprocess=True,
                 *args, **kwargs):
        """
        Verbalizer class
        :param source: List of JSON lines (Thermostat) or path to a local dataset of explanations
        :param model_type: Which model is being explained? optional; will look by itself
        :param standard_samples: how many samples should be loaded if nothing specified
        :param config: config dictionary; optional -> how to will be included
        :param dev: should numerical values for each explanation be returned too?

        TODO warning: object to change:
        --> start of config example <--
        cfg =
        {
        "sgn": "+",                     options: "+", "-", None
        "samples": -1,                  options: {-1, n where n > 0}, n is integer
        "len_filters": 5,               options: n where n > 0, n is integer
        "metric": "mean_sum: 10"        options: see possible metrics
        "dev": True
        }

        possible metrics:
        : "mean_sum: n" where n is a float ranging from 0 to 1;
                                                   uses mean(sum(sorted(attribs[:len * n])) as metric

        TODO    : "quantile: n" where n is a float ranging from 0 to inf
        TODO    : "variance: n : m" where n, m is a float ranging from -inf to inf; n <= m

        -->  end  of config example <--
        """

        self.models = {
            "bert": (r"BertTokenizer", r"bert-base-uncased"),
            "albert": (r"AlbertTokenizer", r"albert-base-v2"),
            "roberta": (r"RobertaTokenizer", r"roberta-base"),
            "xlnet": (r"XLNetTokenizer", r"xlnet-base-cased"),
            "electra": (r"ElectraTokenizer", r"google/electra-small-discriminator")
        }

        if not source:
            raise RuntimeError("Source cannot be empty.")
        self.source = source

        self.model_type = model_type
        if not model_type:
            for model in self.models.keys():
                if model in config['source'].split('-'):
                    self.model_type = model

        if not self.model_type:
            raise RuntimeError("Please specify model_type; Missing param model_type")

        self.standard_samples = standard_samples
        self.modes = ["convolution search", "span search", "compare search", "total order", "compare searches"]
        # which search-algorithms to use
        self.checkpoint = 0  # where did the Verbalizer stop loading examples
        self.len_filters = len_filters
        self.sgn = None
        self.metric = None
        self.label_names = None
        self.dev = dev
        self.multiprocess = multiprocess
        #self.tokenizer = eval("tokenizers.Tokenizer.from_pretrained('{}')"  # force_download=True,
        #                      .format(self.models[self.model_type][1]))
        self.tokenizer = eval("transformers.{}.from_pretrained('{}', cache_dir='data')"  # force_download=True,
                              .format(self.models[self.model_type][0],
                                      self.models[self.model_type][1]))

        if config:
            if config["samples"]:
                self.standard_samples = config["samples"]
            if config["sgn"]:
                self.sgn = config["sgn"]
            if config["metric"]:
                self.metric = config["metric"]
            if config["dev"]:
                self.dev = config["dev"]
                self.maxwords = config["maxwords"]
                self.mincoverage = config["mincoverage"]

    def add_explanations_to_dict(self, data, dct):
        for counter in range(len(data)):
            _ = json.loads(data[counter])
            dct[counter] = {"input_ids":  self.tokenizer.convert_ids_to_tokens(_["input_ids"]),
                            "attributions": _["attributions"],
                            "label": _["label"],
                            "predictions": _["predictions"]
                            }

    def read_samples(self, num_entries: int = None, recursion=0):
        """
        reads num_entries entries in dataset and converts to ordered dict
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
            if type(self.source) == str:
                with open(self.source, "r") as dataset:
                    dataset = list(dataset)
            else:
                dataset = self.source
            self.add_explanations_to_dict(dataset, _dict)

        else:
            self.checkpoint = 0
            if type(self.source) == str:
                with open(self.source, "r") as dataset:
                    dataset_snippet = []
                    for i in range(self.checkpoint):
                        dataset.readline()

                    for i in range(num_entries):
                        dataset_snippet.append(dataset.readline())
                        self.checkpoint += 1
                    #add_explanations_to_dict(dataset_snippet, _dict)
            else:
                dataset_snippet = []
                for i in range(num_entries):
                    dataset_snippet.append(self.source[i+self.checkpoint])
            self.add_explanations_to_dict(dataset_snippet, _dict)

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
                                    "predictions": _dict[sample]["predictions"],
                                    "was_correct": _dict[sample]["label"] == _dict[sample]["predictions"].index(
                                        max(_dict[sample]["predictions"]))}
        """
        if _:
            self.label_names = _["dataset"]["label_names"]
        else:
            self.label_names = dataset_snippet["dataset"]["label_names"]
        """
        #self.label_names = dataset_snippet["dataset"]["label_names"]
        return cleaned_dict

    def doit(self, modes: list = None, n_samples: int = None):
        """
        verbalizes n_samples of dataset
        :param modes: optional: which searches to do on dataset
        :param n_samples: optional: how many samples to verbalize
        :param config: TODO class containing customizations to searches i.e. metrics etc.
        :return: n_samples * (verbalized version of heatmap; by total_order, ...) as array of strings (tuple/list)
        """

        # init standard params
        if not n_samples:
            n_samples = self.standard_samples
        if not modes:
            modes = self.modes
        # end of init

        sample_array = self.read_samples(n_samples)

        explanations = {}
        orders_and_searches = {}
        """
        example of how to use search algorithm
        if "total_order" in modes:
            orders_and_searches["total_order"] = self.total_order(sample_array)
        """
        if self.multiprocess:
            to_calculate = []
            manager = multiprocessing.Manager()
            shared_obj = manager.dict()

            processes = []
            with tqdm(total=len(modes)) as pbar:
                if "convolution search" in modes:
                    conv_proc = multiprocessing.Process(target=svs.shared_memory_convsearch, args=(self.sgn,
                                                                                                   sample_array,
                                                                                                   self.len_filters,
                                                                                                   self.metric,
                                                                                                   shared_obj),
                                                        name="convsearch")
                    processes.append(conv_proc)
                    conv_proc.start()
                    while psutil.virtual_memory().available / (1024**3) < 3:
                        time.sleep(1)
                        print("waiting for memory")

                if "span search" in modes:
                    span_proc = multiprocessing.Process(target=svs.shared_memory_spansearch, args=(self.sgn,
                                                                                                   sample_array,
                                                                                                   self.len_filters,
                                                                                                   self.metric,
                                                                                                   shared_obj),
                                                        name="spansearch")
                    processes.append(span_proc)
                    span_proc.start()
                    while psutil.virtual_memory().available / (1024**3) < 3:
                        time.sleep(1)
                        print("waiting for memory")

                for i in processes:
                    print("\nwaiting for", i.name)
                    i.join()
                print("calculations done\nwaiting for data collection")

                if "convolution search" in modes:
                    explanations["convolution search"] = shared_obj["convolution search"][0]
                    orders_and_searches["convolution search"] = shared_obj["convolution search"][1]
                    pbar.update(1)

                if "span search" in modes:
                    explanations["span search"] = shared_obj["span search"][0]
                    orders_and_searches["span search"] = shared_obj["span search"][1]
                    pbar.update(1)

                processes = []
                print("collection done\nwaiting for compare search")

                if "compare search" in modes:
                    explanations["compare search"] = sm.compare_search(orders_and_searches, sample_array)
                    pbar.update(1)
                print("compare search done\nwaiting for ordering")
                if "total order" in modes:  # NO PROCESS because fast
                    explanations["total order"] = t.verbalize_total_order(t.total_order(sample_array))
                    pbar.update(1)

                print("ordering done\nwaiting for search concatenation")
                if "compare searches" in modes:
                    memory = psutil.virtual_memory().available / (1024**3)
                    spareable_mem = memory * 0.66
                    amount_procs = min(int(spareable_mem/3), int(psutil.cpu_count()))

                    chunkssize = int(len(sample_array.keys())/amount_procs)
                    sample_keys = list(sample_array.keys())
                    sample_keys_chunks_points = []
                    for i in range(0, amount_procs):
                        sample_keys_chunks_points.append((i+1)*chunkssize+1)
                    #TODO add mp support for compare_searches
                    explanations["compare searches"] = t.compare_searches(orders_and_searches, sample_array)
                    pbar.update(1)
                    print("search concatenation done\nnow verbalizing")

            if not self.dev:
                return explanations, sample_array, None
            else:
                return explanations, sample_array, orders_and_searches


        #
        #
        #
        #
        else:
            with tqdm(total=len(modes)) as pbar:
                if "convolution search" in modes:
                    if not self.sgn:
                        (explanations["convolution search"], orders_and_searches["convolution search"]) = sm.convolution_search(
                            sample_array.copy(), self.len_filters, metric=self.metric)
                    else:
                        (explanations["convolution search"], orders_and_searches["convolution search"]) = sm.convolution_search(
                            sample_array.copy(), self.len_filters, self.sgn, self.metric)
                pbar.update(1)
                if "span search" in modes:
                    if not self.sgn:
                        (explanations["span search"], orders_and_searches["span search"]) = sm.span_search(
                            sample_array, self.len_filters, metric=self.metric)
                    else:
                        (explanations["span search"], orders_and_searches["span search"]) = sm.span_search(
                            sample_array, self.len_filters, self.sgn, self.metric)
                pbar.update(1)
                # SHOULD ALWAYS BE DONE AT THE END but before total search
                if "compare search":
                    explanations["compare search"] = sm.compare_search(orders_and_searches, sample_array)
                pbar.update(1)
                if "total order" in modes:
                    explanations["total order"] = t.verbalize_total_order(t.total_order(sample_array))
                pbar.update(1)
                if "compare searches" in modes:
                    explanations["compare searches"] = t.compare_searches(orders_and_searches, sample_array)
                pbar.update(1)
            # TODO: Maybe detokenize input_ids using tokenizer from self?
        if not self.dev:
            return explanations, sample_array, None
        else:
            return explanations, sample_array, orders_and_searches

    def __call__(self, modes: list = None, n_samples: int = None, *args, **kwargs):
        return self.doit(modes, n_samples)

    def filter_verbalizations(self, verbalizations, samples, orders_and_searches, maxwords=100, mincoverage=.1, *args):
        """

        :param verbalizations: takes output[0] of self.doit()
        :param samples: takes output[1] of self.doit()
        :param orders_and_searches: requires self.dev enabled and output[2] of self.doit()
        :param maxwords: maximum words in sample to be returned
        :param mincoverage: minimum coverage needed for a sample to be returned (0.0 - 1.0)
        :return: filtered verbalizations
        """
        tofilter = self.modes[:len(self.modes)-3]  # MANUAL NEEDS TO BE CHANGED WITH EVERY ADDED SEARCHTYPE
        filter_len = lambda x: [len(x[i]["input_ids"]) < maxwords for i in x.keys()]
        filter_verbalizations = lambda x, n, searchtype: [sum(samples[n]["attributions"][min(i):max(i)]) /
                                                          sum(samples[n]["attributions"]) > mincoverage
                                                          for i in x[searchtype][n]["indices"]]
        valid_indices = filter_len(samples)
        valid_indices_ = []
        for sampleindex in samples.keys():
            if valid_indices[sampleindex]:
                for search_type in tofilter:
                    try:
                        if any(filter_verbalizations(orders_and_searches, sampleindex, search_type)):
                            valid_indices_.append(sampleindex)
                            break
                    except TypeError:
                        pass  # no valid snippet existed in the first place resulting in no value to check thus -> error
        return valid_indices_
