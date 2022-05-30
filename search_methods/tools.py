import numpy as np
from numba import jit


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