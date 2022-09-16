from src.search_methods.tools import *
import numpy as np
import src.search_methods.filters as fl
# span based filter search


def span_search(samples: dict, filter_length, top_n_coherences: int = 5, sgn=None, mode: dict = None,
                randomize_attribs=False):
    """
    generates spans to search for coherences amongst a sample loaded by dataloader.Verbalizer.read_samples()
    works like field_search(*args)
    only used for calculations; return is to be used for verbalization
    :param samples: shape = dataloader.Verbalizer.read_samples() return shape
    :param filter_length: length of generated filters; computationally expensive, generates filter_length^2 filters
    :param top_n_coherences: how many coherences should be returned?
    :param sgn: sign sensitive search?
    :param mode: metric to use
    :param randomize_attribs: do we want to randomize the sample attributions
    :return: dictionary composed of: keys=keys of samples values: (sample_snippet_indices, sample_snippet_values)
    """

    sorted_filters = generate_spans(filter_length)
    words_and_vals = {}
    for key in samples.keys():
        sample = samples[key]
        coherent_words_sum, coherent_values_sum = fl.single_convolution_search(sample, sgn, mode, sorted_filters,
                                                                               randomize_attribs)
        _words, _values = fl.result_filtering(coherent_words_sum, coherent_values_sum)
        words_and_vals[key] = {"indices": _words,
                               "values": _values}

    return words_and_vals  # return as dictionary_ [key] = tuple(word_indices, word/snippet_values)


def generate_spans(filter_length):
    """
    generates spans of size filter_length with varying amount of ones for filter_length=5:
    [0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1]
    :param filter_length: size of spans
    :return: list of np-byte arrays with binary spans to be used in tools.filter_span_sample_sum()
    """
    filters = []
    for i in range(1, filter_length+1):
        if i % 2 == 0:
            continue
        num_zeros = filter_length - i
        num_ones = i
        span = [*[0]*(int(num_zeros/2)),
                *[1]*num_ones,
                *[0]*int(num_zeros/2)]
        if sum(span) > 1:
            filters.append(span)

    filters = np.array(filters).astype("byte")
    return filters

# end of span based filter search
