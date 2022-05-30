import warnings
from warnings import warn
import numpy as np
from numba import jit  # weiß nicht ob wichtig
from itertools import permutations
from tqdm import tqdm as tqdm
warnings.simplefilter("always", ResourceWarning)
warnings.simplefilter("always", DeprecationWarning)


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

# end of order

# Filters and Spans #

# start of filter-based search


def field_search(samples: dict, filter_length, top_n_coherences: int = 5):
    """
    perfoms coherency search amongst a sample loaded by dataloader.Verbalizer.read_samples()
    first generates binary-filters of length n (for example [1, 0, 1], [0, 1, 1] or [1, 1, 0]
    then filters the sample with the generated filters
    :param samples: shape = dataloader.Verbalizer.read_samples() return shape
    :param filter_length: length of generated filters; computationally expensive, generates filter_length^2 filters
    :param top_n_coherences: how many coherences should be returned?
    :return: verbalized search:string
    """

    if filter_length > 8:
        warn("Filter length of >8 is not recommended, as it is computationally very"
             " expensive and unlikely to give results",
             category=ResourceWarning)

    # indices = np.arange(0, len(sample["attributions"])).astype("uint16")

    sorted_filters = generate_filters(filter_length)
    verbalizations = []
    for key in tqdm(samples.keys()):
        sample = samples[key]
        attribs = np.array(sample["attributions"]).astype("float32")
        attribs = attribs/abs(np.max(attribs))
        metric = sorted(attribs)
        metric = sum(metric[:10])/10
        coherent_words_sum, coherent_values_sum = filter_span_sample_sum(sorted_filters, attribs, metric)

        coherent_words_sum, coherent_values_sum = zip(*reversed(sorted(zip(coherent_words_sum, coherent_values_sum))))
        _words = []
        _values = []
        for i in range(len(coherent_words_sum)):
            if not coherent_words_sum[i] in _words:
                _words.append(coherent_words_sum[i])
                _values.append(coherent_values_sum[i])

        coherent_words_sum = _words
        coherent_values_sum_sum = _values

        verbalization = ""
        for filter_result in coherent_words_sum[:top_n_coherences]:
            for input_id in filter_result:
                verbalization += sample["input_ids"][input_id] + ", "
            verbalization += "are related to each other \n"
        verbalizations.append(verbalization)

    return verbalizations


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


# end of filter based search

# span based filter search

def span_search(samples: dict, filter_length, top_n_coherences: int = 5):
    """
    generates spans to search for coherences amongst a sample loaded by dataloader.Verbalizer.read_samples()
    works like field_search(*args)
    :param samples: shape = dataloader.Verbalizer.read_samples() return shape
    :param filter_length: length of generated filters; computationally expensive, generates filter_length^2 filters
    :param top_n_coherences: how many coherences should be returned?
    :return: verbalized search:string
    """

    # indices = np.arange(0, len(sample["attributions"])).astype("uint16")

    sorted_filters = generate_spans(filter_length)
    verbalizations = []
    for key in tqdm(samples.keys()):
        sample = samples[key]
        attribs = np.array(sample["attributions"]).astype("float32")
        attribs = attribs/abs(np.max(attribs))
        metric = sorted(attribs)
        metric = sum(metric[:10])/10
        coherent_words_sum, coherent_values_sum = filter_span_sample_sum(sorted_filters, attribs, metric)

        coherent_words_sum, coherent_values_sum = zip(*reversed(sorted(zip(coherent_words_sum, coherent_values_sum))))
        _words = []
        _values = []
        for i in range(len(coherent_words_sum)):
            if not coherent_words_sum[i] in _words:
                _words.append(coherent_words_sum[i])
                _values.append(coherent_values_sum[i])

        coherent_words_sum = _words
        coherent_values_sum_sum = _values

        verbalization = ""
        for filter_result in coherent_words_sum[:top_n_coherences]:
            for input_id in filter_result:
                verbalization += sample["input_ids"][input_id] + ", "
            verbalization += "are related to each other \n"
        verbalizations.append(verbalization)

    return verbalizations


def generate_spans(filter_length):
    filters = []
    for i in range(filter_length):
        filters.append(np.ones(i).astype("byte"))

    return filters

# end of span based filter search


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
    for index in range(len(coherency_values)):
        if coherency_values[index] > np.median(coherency_values):
            coherent_snippets.append(coherent_words[index])

    return coherent_snippets, coherency_values

# End of Filters and Spans #


# metrics #


@jit(nopython=True)
def get_variance(attribs):
    """
    variance
    :param attribs: 1D matrix
    :return: variances of each value in matrix
    """
    expected_value = np.mean(attribs)
    variances = []

    for value in attribs:
        variances.append(((expected_value - value)**2))

    return variances


@jit(nopython=True)
def get_stdev(variances):
    """
    standard deviance
    :param variances: array of variances of a given 1D matrix
    :return: standard deviances of variances
    """
    stdevs = []

    for value in variances:
        stdevs.append(value**.5)

    return stdevs


# end of metrics #
