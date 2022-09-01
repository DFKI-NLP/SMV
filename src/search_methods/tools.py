import json
import os
import yaml

import datasets
import numpy as np
import pandas as pd
import thermostat

from collections import defaultdict
from numba import jit
from tqdm import tqdm
from typing import List

from src import dataloader
from vis import color_str, Color


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
    coherent_snippets = []

    return coherent_words, coherency_values


# order input sample using >= :
def total_order(_dict: dict):
    """
    takes a ordered dict and performs total order >= on it; doesnt change original dict
    :param _dict: dictionary of shape @return of Verbalizer.read_samples(args)
    :return: dict of shape {index_by_magnitude_of_value : (token : value)}
    """

    # in order to sort efficiently we need to do a little spaghetti code
    # idea:
    # we first sort the attributions using precompiled sorted and then change the input ids accordingly
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


def verbalize_total_order(ordered_dict):
    verbalizations = {}
    for sample in ordered_dict.keys():
        sum_vals = 0
        for attrib in ordered_dict[sample]["attributions"]:
            sum_vals += attrib if attrib > 0 else 0
        verbalization = ["top tokens are:"]
        for token in range(len(ordered_dict[sample]["input_ids"])):
            if sum_vals != 0:
                prediction_score = round(100*ordered_dict[sample]["attributions"][token] / sum_vals, 2)
                verbalization.append("token '" + ordered_dict[sample]["input_ids"][token].replace("▁", " ")
                                     + "' with {}% of prediction score"
                                     .format(prediction_score))
            else:
                verbalization.append("token '" + ordered_dict[sample]["input_ids"][token].replace("▁", " ")
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
    variance
    :param attribs: 1D matrix
    :return: variances of each value in matrix
    """
    expected_value = np.mean(attribs)
    variance = sum((-attribs + expected_value)**2)/len(attribs)
    return variance


@jit(nopython=True)
def get_stdev(variance):
    """
    standard deviation
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
    args = [selected_mode[1](mode["value"])]  # FIXME: In which cases do we need multiple args?
    #args = [selected_mode[1](_args[i]) for i in range(1, len(_args))]
    return args


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
        sum_values = 0
        for i in samples[key]["attributions"]:
            sum_values += i if i > 0 else 0

        words = []
        values = []
        sorted_by_max_values = [i for i in reversed(sorted(prepared_data[key]["values"]))]
        for value in sorted_by_max_values:
            indexof = prepared_data[key]["values"].index(value)
            try:
                values.append(sum([samples[key]["attributions"][indexid]
                                   for indexid in prepared_data[key]["indices"][indexof]]))
                _ = []
                for entry in prepared_data[key]["indices"][indexof]:
                    _.append(samples[key]["input_ids"][entry]) #.replace("▁", " "))
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
            except Exception as e:
                verbalization = "No coherent values found."
                verb_cov_tuple = (verbalization, 0.0)

            verbalizations.append(verb_cov_tuple)
        verbalization_dict[key] = [v for v, c in sorted(verbalizations, key=lambda vc: vc[1], reverse=True)]
    return verbalization_dict


def compare_search(searches: dict, samples):
    """

    :param searches:
    :param samples:
    :return:
    """
    search_types = searches.keys()
    coincidences = {}    
    for subclass in search_types:
        for subclass_2 in search_types:
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


def compare_searches(searches: dict, samples):
    """

    :param searches:
    :param samples:
    :return:
    """

    # search_types = searches.keys() ##UNUSED##

    # sample_info = []  ##UNUSED##
    verbalized_explanations = {}
    for sample_key in tqdm(searches[list(searches.keys())[0]].keys()):
        sample_atts = samples[sample_key]["attributions"]
        input_ids = samples[sample_key]["input_ids"]
        candidates = defaultdict(dict)

        for stype in list(searches.keys()):
            explore_search(candidates, stype, searches, sample_key, sample_atts)

        for i, attr in enumerate(sample_atts):
            candidates['total search'][str(i)] = coverage([i], sample_atts)

        conv_top5 = sorted(candidates['convolution search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]
        span_top5 = sorted(candidates['span search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]
        total_top5 = sorted(candidates['total search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]

        combined_candidate_indices = []

        combine_results(conv_top5, combined_candidate_indices)
        combine_results(span_top5, combined_candidate_indices)
        combine_results(total_top5, combined_candidate_indices)

        final_spans = []
        for i in sorted(combined_candidate_indices):
            if len(final_spans) > 0 and final_spans[-1][-1] + 1 == i:
                final_spans[-1].append(i)
            else:
                final_spans.append([i])

        cov_fs = []
        for fs in final_spans:
            cov_fs.append(coverage(fs, sample_atts))
        upper_quartile = np.quantile(cov_fs, 0.75)

        num_uq_spans = len([cov_fs[i] > upper_quartile for i, fs in enumerate(final_spans)])
        spans_with_ranks = {}
        for i, fs in enumerate(final_spans):
            rank = sorted(cov_fs, reverse=True).index(cov_fs[i])

            if cov_fs[i] < upper_quartile:
                if num_uq_spans == 0 and rank == 0:
                    pass
                else:
                    continue

            if len(fs) == 1:
                token = input_ids[fs[0]].replace('Ġ', '')
                verbalization = f"The word » {token} «"
            else:
                span = " ".join([input_ids[t] for t in fs]).replace(' ', '').replace('Ġ', ' ').replace(
                    '<s>', '').replace('</s>', '').replace('<pad>', '')
                if "." in span:
                    verbalization = f"The span » {span} «"
                else:
                    verbalization = f"The phrase » {span} «"

            if rank == 0:
                verbalization += " is most important for the prediction"
            else:
                verbalization += " is also salient"

            cov_str = str(round(100 * cov_fs[i]))
            if cov_str == "0":
                continue
            verbalization += " (" + cov_str + " %)."

            spans_with_ranks[rank] = verbalization

        if len(spans_with_ranks) == 0:
            continue
        ranked_spans = sorted(spans_with_ranks.items(), key=lambda k_v: k_v[0])

        # TODO: The span.replace(...) has to be different for other models/tokenizers
        # BERT
        # verbalized_explanations[sample_key] = " ".join([span.replace(' ##', '') for i, span in ranked_spans])

        # RoBERTa
        verbalized_explanations[sample_key] = " ".join([span for i, span in ranked_spans])
        # sample_info.append(samples[sample_key])
        return verbalized_explanations


@jit(nopython=True)
def explore_search(candidates, search_type, searches, sample_key, sample_atts):
    candidates[search_type] = {}
    for indices in searches[search_type][sample_key]["indices"]:
        candidates[search_type][','.join([str(idx) for idx in indices])] = coverage(indices, sample_atts)
    return candidates


@jit(nopython=True)
def coverage(span, attributions):
    if span[0]:
        pos_att_sum = sum([float(a) if a > 0 else 0 for a in attributions])
        if pos_att_sum > 0:
            return sum([attributions[w] for w in span]) / pos_att_sum
    return 0


@jit(nopython=True)
def combine_results(result_dict, combined_candidate_indices):
    for idx_cov_tuple in result_dict:
        if ',' in idx_cov_tuple[0]:
            indices = idx_cov_tuple[0].split(',')
        else:
            indices = [idx_cov_tuple[0]]
        for idx in indices:
            if idx == 'None':
                continue
            if int(idx) not in combined_candidate_indices:
                combined_candidate_indices.append(int(idx))


def get_binary_attributions_from_annotator_rationales(text: str, rationales: List[str]):
    # TODO
    """ Helper function to turn the text spans of annotator rationales (e.g. from the Movie Rationales dataset) into
    binary attributions"""
    binary_attributions = []

