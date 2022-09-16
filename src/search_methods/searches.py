from warnings import warn
from src.search_methods.tools import *
from itertools import permutations
from typing import Tuple, List, Dict
# start of filter-based search
#warnings.simplefilter("always")


def convolution_search(samples: dict, filter_length, top_n_coherences: int = 5, sgn=None, mode: str = "mean: 1",
                       randomize_attribs=False) -> dict:
    # split for multiple processes -> per instance calculation
    """
    perfoms coherency search amongst a sample loaded by dataloader.Verbalizer.read_samples()
    first generates binary-filters of length n (for example [1, 0, 1], [0, 1, 1] or [1, 1, 0]
    then filters the sample with the generated filters
    :param samples: shape = dataloader.Verbalizer.read_samples() return shape
    :param filter_length: length of generated filters; computationally expensive    , generates filter_length^2 filters
    :param top_n_coherences: how many coherences should be returned?
    :param sgn: sign sensitive search?
    :param mode: metric to use
    :param randomize_attribs: Fill attributions with random values for baseline in human evaluation
    :return: verbalized search:string
    """
    if filter_length > 8:
        warn("Filter length of >8 is not recommended, as it is computationally very"
             " expensive and unlikely to give results",
             category=ResourceWarning)

    # indices = np.arange(0, len(sample["attributions"])).astype("uint16")

    sorted_filters = generate_filters(filter_length)
    words_and_vals = {}
    for key in samples.keys():
        sample = samples[key]
        coherent_words_sum, coherent_values_sum = single_convolution_search(sample, sgn, mode,
                                                                            sorted_filters, randomize_attribs)
        # issue: how to keep track of samples
        _words, _values = result_filtering(coherent_words_sum, coherent_values_sum)
        words_and_vals[key] = {"indices": _words,
                               "values": _values}

    return words_and_vals  # return as dictionary_ [key] = tuple(word_indices, word/snippet_values)


def generate_filters(filter_length):
    """
    generates binary-search filters
    :param filter_length: max length of filter
    :return: filter_length^2 filters with all permutations
    """
    filters = []
    for i in range(filter_length):
        if 1 < i < filter_length - 1:
            filters.append([*[*[1] * i, *[0]*(filter_length-i)]])
    filters = permute_filter_blueprints(filters)
    return filters


def single_convolution_search(sample: dict, sgn: str,
                              mode: any,
                              sorted_filters: np.ndarray,
                              randomize_attribs: bool = False) -> Tuple[List, List]:
    attribs = np.array(sample["attributions"]).astype("float32")/abs(np.max(sample["attributions"]))  # normalized

    if randomize_attribs:
        attribs = np.random.rand(*attribs.shape)

    if "mean" in mode["name"]:
        metric = get_mean(get_metric_values(mode)[0:], attribs)

    elif "quantile" in mode["name"]:
        stdevval = get_stdev(get_variance(attribs))
        metric = stdevval * get_metric_values(mode)[0]

    elif "variance" in mode["name"]:
        variance = get_variance(attribs)
        metric = variance * get_metric_values(mode)[0]
    else:
        raise RuntimeError("no metric specified")

    try:
        if not sgn:
            coherent_words_sum, coherent_values_sum = filter_span_sample_sum(sorted_filters, attribs, metric)
        else:
            coherent_words_sum, coherent_values_sum = filter_span_sample_sum_sgn(sorted_filters, attribs, metric,
                                                                                 sgn)
    except Exception as e:
        coherent_words_sum, coherent_values_sum = [[None]], [[None]]

    coherent_words_sum, coherent_values_sum = zip(*reversed(sorted(zip(coherent_words_sum, coherent_values_sum))))
    return coherent_words_sum, coherent_values_sum


def permute_filter_blueprints(filters):
    """
    permutes a given filter to all possible combinations
    :param filters: binary filter
    :return: (sum(filter) over len(filter)) filters (binomial coefficient)
             (all possible combinations of the 0s and 1s in filter)
    """
    permuted_filters = []
    for i in filters:
        permuted_filters.append(sorted(tuple(set(permutations(i)))))

    filters = []
    for i in permuted_filters:
        for j in i:
            filters.append(j)
    filters = np.array(filters).astype("byte")
    return filters


def generate_filters(filter_length):
    """
    generates binary-search filters
    :param filter_length: max length of filter
    :return: filter_length^2 filters with all permutations
    """
    filters = []
    for i in range(filter_length):
        if 1 < i < filter_length - 1:
            filters.append([*[*[1] * i, *[0]*(filter_length-i)]])
    filters = permute_filter_blueprints(filters)
    return filters


def result_filtering(coherent_words_sum, coherent_values_sum):
    _words = []
    _values = []
    for i in range(len(coherent_words_sum)):
        if not coherent_words_sum[i] in _words:
            _words.append(coherent_words_sum[i])
            _values.append(coherent_values_sum[i])

    return _words, _values


def permute_filter_blueprints(filters):
    """
    permutes a given filter to all possible combinations
    :param filters: binary filter
    :return: (sum(filter) over len(filter)) filters (binomial coefficient)
             (all possible combinations of the 0s and 1s in filter)
    """
    permuted_filters = []
    for i in filters:
        permuted_filters.append(sorted(tuple(set(permutations(i)))))

    filters = []
    for i in permuted_filters:
        for j in i:
            filters.append(j)
    filters = np.array(filters).astype("byte")
    return filters


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
        coherent_words_sum, coherent_values_sum = single_convolution_search(sample, sgn, mode, sorted_filters,
                                                                               randomize_attribs)
        _words, _values = result_filtering(coherent_words_sum, coherent_values_sum)
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
# end of filter based search
