import os
import yaml
from typing import Tuple

import datasets
import numpy as np
import thermostat

import src.fastcfg as fc

from numba import jit
from typing import List


@jit(nopython=True)
def filter_span_sample_sum(sorted_filters, sample_attribs, metric_value):
    """
    use binary filters on a 1 dimensional matrix
    sum up the product of filter x snippet
    compare the sum to a metric value
    if sum > metric value:
        return sum, values
    :param sorted_filters: binary filters
    :param sample_attribs: 1D matrix
    :param metric_value: threshold
    :return: list of possibly coherent samples, corresponding values
    """
    index_offset = len(sorted_filters[0])
    coherent_words = []
    coherency_values = []
    for filter_ in sorted_filters:
        for index in range(len(sample_attribs) - index_offset):
            coherency_value = sum(filter_ * sample_attribs[index:index+index_offset]) / sum(filter_)
            if coherency_value > metric_value:
                coherent_words.append([index + i for i in range(len(filter_)) if filter_[i] != 0])
                coherency_values.append([coherency_value])

    coherency_values = np.array(coherency_values)
    coherent_snippets = []
    return coherent_words, coherency_values


@jit(nopython=True)
def filter_span_sample_sum_sgn(sorted_filters, sample_attribs, metric_value, sgn: str = "+"):
    """
    use binary filters on a 1 dimensional matrix
    sum up the product of filter x snippet
    compare the sum to a metric value
    if sum > metric value:
        return sum, values
    :param sorted_filters: binary filters
    :param sample_attribs: 1D matrix
    :param metric_value: threshold
    :param sgn: which sign to look at; +-> vals => 0; - vals < 0
    :return: list of possibly coherent samples, corresponding values
    """
    index_offset = len(sorted_filters[0])
    coherent_words = []
    coherency_values = []
    for filter_ in sorted_filters:
        for index in range(len(sample_attribs) - index_offset):
            filter_result = filter_ * sample_attribs[index:index+index_offset]
            sgn_consistent = True
            if sgn == "+":
                for val in filter_result:
                    if val >= 0:
                        pass
                    else:
                        sgn_consistent = False

            elif sgn == "-":
                for val in filter_result:
                    if val < 0:
                        pass
                    else:
                        sgn_consistent = False

            if sgn_consistent:
                coherency_value = sum(filter_result) / sum(filter_)
                if coherency_value > metric_value:
                    coherent_words.append([index + i for i in range(len(filter_)) if filter_[i] != 0])
                    coherency_values.append([coherency_value])

    coherency_values = np.array(coherency_values)
    return coherent_words, coherency_values


# order input sample using >= :
def total_order(_dict: dict) -> dict:
    """
    takes a ordered dict and performs total order >= on it; returns ordered dict, doesnt do inplace ordering
    :param _dict: dictionary of shape @return of Verbalizer.read_samples(args)
    :return: dict of shape {index_by_magnitude_of_value : (token : value)}
    """

    sorted_dict = {}
    for sample in _dict.keys():
        ids = _dict[sample]["input_ids"]
        _attributions = _dict[sample]["attributions"]

        attributions = {}
        for i in range(len(_attributions)):
            attributions[i] = _attributions[i]

        sorted_ids = []
        attributions = sorted(attributions.items(), key=lambda x: x[1])
        for pair in attributions:
            sorted_ids.append(ids[pair[0]])

        sorted_dict[sample] = {"input_ids": [i for i in reversed(sorted_ids)],
                               "attributions": [i[1] for i in reversed(attributions)],
                               "label": _dict[sample]["label"],
                               "predictions": _dict[sample]["predictions"]
                               }

    return sorted_dict


def verbalize_total_order(ordered_dict: dict) -> dict:
    """
    returns verbal representation of total order
    :param ordered_dict: dictionary like see total_order's return
    :return: dict, containing verbalized explanations
    """
    verbalizations = {}
    for sample in ordered_dict.keys():
        sum_vals = 0
        for attrib in ordered_dict[sample]["attributions"]:
            sum_vals += attrib if attrib > 0 else 0
        verbalization = []
        for token in range(len(ordered_dict[sample]["input_ids"])):
            if sum_vals != 0:
                prediction_score = round(100*ordered_dict[sample]["attributions"][token] / sum_vals, 2)
                verbalization.append("Token '" + ordered_dict[sample]["input_ids"][token].replace("▁", " ")
                                     + "' with {}% of prediction score"
                                     .format(prediction_score))
            else:
                verbalization.append("Token '" + ordered_dict[sample]["input_ids"][token].replace("▁", " ")
                                     + "' with {}% of prediction score"
                                     .format("NaN"))
        verbalizations[sample] = verbalization

    return verbalizations

# end of order


# metrics #
def get_mean(num_vals, attribs):
    """
    returns mean of top-%num_vals values of attribs
    :param num_vals: integer determining how many values to be used
    :param attribs: 1D array of any numbers
    :return: float- mean of top values
    """
    mean = [*reversed(sorted(attribs))]
    mean = sum(mean[:int(len(mean)*num_vals[0])]) / int(len(mean)*num_vals[0])
    return mean


@jit(nopython=True)
def get_variance(attribs):
    """
    variance of array
    :param attribs: 1D matrix
    :return: variances of each value in matrix
    """
    expected_value = np.mean(attribs)
    variance = sum((-attribs + expected_value)**2)/len(attribs)
    return variance


@jit(nopython=True)
def get_stdev(variance):
    """
    standard deviation if array
    :param variances: array of variances of a given 1D matrix
    :return: standard deviances of variances
    """
    stdevs = variance**.5

    return stdevs


# end of metrics #

# utitlity functions
def get_metric_values(mode: dict):
    """
    Turns config - mode for metric into processable format
    :param mode: dictionary containing name and value
    :return: params for actual metric
    """
    modes = {
        "mean": (1, float),
        "quantile": (1, float),
        "variance": (2, float),
        "foo": NotImplementedError,
    }
    selected_mode = modes[mode["name"]]
    args = [selected_mode[1](mode["value"])]
    return args


def single_verbalize_field_span_search(prepared_data, samples, sgn="+"):
    """
    Verbalizes results of field_search or span_search
    :param prepared_data: output of one of the searches
    :param samples: input array -> see dataloader.read_samples
    :param sgn: unused for now
    :return: Dict[sample_key: verbalization_list->list ]
    """

    sum_values = 0
    for i in samples["attributions"]:
        sum_values += i if i > 0 else 0

    words = []
    values = []
    sorted_by_max_values = [i for i in reversed(sorted(prepared_data["values"]))]
    for value in sorted_by_max_values:
        indexof = prepared_data["values"].index(value)
        try:
            values.append(sum([samples["attributions"][indexid]
                               for indexid in prepared_data["indices"][indexof]]))
            _ = []
            for entry in prepared_data["indices"][indexof]:
                _.append(samples["input_ids"][entry]) #.replace("▁", " "))
            words.append(_)
        except TypeError:
            words.append([None])

    verbalizations = []
    for snippet in range(len(words)):
        verbalization = "snippet: '"
        for word in words[snippet]:
            if not word:
                continue
            verbalization += word + " "
        try:
            coverage = round((values[snippet] / sum_values) * 100, 2)
            verbalization += "' contains {}% of prediction score.".format(str(coverage))
            verb_cov_tuple = (verbalization, coverage)
        except Exception as e:  # Out of convenience, keep this unspecified.
            verbalization = "No coherent values found."
            verb_cov_tuple = (verbalization, 0.0)

        verbalizations.append(verb_cov_tuple)
    return verbalizations


def verbalize_field_span_search(prepared_data, samples, sgn="+"):
    """
    Verbalizes results of field_search or span_search
    :param prepared_data: output of one of the searches
    :param samples: input array -> see dataloader.read_samples
    :param sgn: unused for now
    :return: Dict[sample_key: verbalization_list->list ]
    """
    verbalization_dict = {}
    for key in prepared_data.keys():
        verbalizations = single_verbalize_field_span_search(prepared_data[key], samples[key], sgn=sgn)
        verbalization_dict[key] = [v for v, c in sorted(verbalizations, key=lambda vc: vc[1], reverse=True)]
    return verbalization_dict


def compare_search(searches: dict, samples):
    """

    :param searches:
    :param samples:
    :return:
    """
    valid_types = ["span search", "convolution search"]
    coincidences = {}    
    for subclass in valid_types:
        for subclass_2 in valid_types:
            if subclass == subclass_2:
                pass
            else:
                for sample_key in searches[subclass].keys():
                    sum_values = 0
                    for i in samples[sample_key]["attributions"]:
                        sum_values += i if i > 0 else 0

                    _ = []
                    for value_1 in searches[subclass][sample_key]["indices"]:
                        for value_2 in searches[subclass_2][sample_key]["indices"]:
                            if value_1 is None or value_2 is None:
                                continue
                            if value_1 == value_2 and value_1 not in coincidences.items():
                                _.append(value_1)

                    verbalizations = []
                    for snippet in _:
                        verbalization = "snippet: '"
                        snippet_tokens = []
                        for word_index in snippet:
                            if word_index is not None:
                                snippet_tokens.append(samples[sample_key]["input_ids"][word_index]) #.replace("▁", " ")
                        verbalization += ' '.join(snippet_tokens)
                        try:
                            verbalization += "' occurs in all searches and accounts for {}% of prediction score".format(
                                str(round(
                                    (sum([samples[sample_key]["attributions"][i] for i in snippet])/sum_values)*100, 2
                                )))
                        except Exception as e:
                            pass

                        verbalizations.append(verbalization)
                    if not verbalizations:
                        verbalizations = ["No snippet occurs in all searches simultaneously"]
                    coincidences[sample_key] = verbalizations

    return coincidences


def get_binary_attributions_from_annotator_rationales(text: str, rationales: List[str]):
    # TODO
    """ Helper function to turn the text spans of annotator rationales (e.g. from the Movie Rationales dataset) into
    binary attributions"""
    binary_attributions = []
    raise NotImplementedError


def read_config(cfg_path_or_object) -> Tuple[dict, str]:
    config = None
    if not type(cfg_path_or_object) == fc.Config:
        with open(cfg_path_or_object) as stream:
            config = yaml.safe_load(stream)
    else:
        config = yaml.safe_load(cfg_path_or_object())
    if not os.path.isfile(config["source"]):
        if "thermostat/" in config["source"]:
            # Load source from Thermostat configuration
            thermo_config_name = config["source"].replace("thermostat/", "")
            thermo_config = thermostat.load(thermo_config_name, cache_dir="data")
            # Convert to pandas DataFrame and then to JSON lines
            source = thermo_config.to_pandas().to_json(orient="records", lines=True).splitlines()
        elif "datasets/" in config["source"]:
            # TODO: Add annotator rationale datasets
            # https://huggingface.co/datasets/movie_rationales
            # http://www.eraserbenchmark.com/ (BoolQ)
            # Load source from Hugging Face datasets library
            hf_dataset_name = config["source"].replace("datasets/", "")
            hf_dataset = datasets.load_dataset(hf_dataset_name, cache_dir="data")
            raise NotImplementedError("HF datasets not supported atm.")
        else:
            raise NotImplementedError("Invalid source!")
    else:
        source = config["source"]

    return config, source
