from search_methods.tools import *
import numpy as np

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
    for key in samples.keys():
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
    for i in range(1, filter_length+1):
        if i % 2 == 0:
            continue
        num_zeros = filter_length - i
        num_ones = i
        filters.append([*[0]*(int(num_zeros/2)),
                        *[1]*(num_ones),
                        *[0]*int(num_zeros/2)])

    print(filters)
    filters = np.array(filters).astype("byte")
    return filters

# end of span based filter search