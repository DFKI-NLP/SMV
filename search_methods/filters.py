from warnings import warn
from search_methods.tools import *
from itertools import permutations
# start of filter-based search


def field_search(samples: dict, filter_length, top_n_coherences: int = 5, sgn=None, mode: str = "mean: 1"):
    """
    perfoms coherency search amongst a sample loaded by dataloader.Verbalizer.read_samples()
    first generates binary-filters of length n (for example [1, 0, 1], [0, 1, 1] or [1, 1, 0]
    then filters the sample with the generated filters
    :param samples: shape = dataloader.Verbalizer.read_samples() return shape
    :param filter_length: length of generated filters; computationally expensive, generates filter_length^2 filters
    :param top_n_coherences: how many coherences should be returned?
    :param sgn: sign sensitive search?
    :param mode: metric to use
    :return: verbalized search:string
    """

    if filter_length > 8:
        warn("Filter length of >8 is not recommended, as it is computationally very"
             " expensive and unlikely to give results",
             category=ResourceWarning)

    # indices = np.arange(0, len(sample["attributions"])).astype("uint16")

    sorted_filters = generate_filters(filter_length)
    verbalizations = []
    for key in samples.keys():
        sample = samples[key]
        attribs = np.array(sample["attributions"]).astype("float32")/abs(np.max(sample["attributions"]))#normalized
        if "mean" in mode:
            metric = get_mean(get_metric_values(mode)[0:], attribs)

        elif "quantile" in mode:
            metric = get_stdev(get_variance(attribs))

        elif "variance" in mode:
            metric = get_variance(attribs)

        if not sgn:
            coherent_words_sum, coherent_values_sum = filter_span_sample_sum(sorted_filters, attribs, metric)
        else:
            coherent_words_sum, coherent_values_sum = filter_span_sample_sum_sgn(sorted_filters, attribs, metric, sgn)

        coherent_words_sum, coherent_values_sum = zip(*reversed(sorted(zip(coherent_words_sum, coherent_values_sum))))
        _words = []
        _values = []
        for i in range(len(coherent_words_sum)):
            if not coherent_words_sum[i] in _words:
                _words.append(coherent_words_sum[i])
                _values.append(coherent_values_sum[i])

        coherent_words_sum = _words
        coherent_values_sum = _values



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