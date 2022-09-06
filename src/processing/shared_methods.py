from src.search_methods import tools as t, spans as span, filters as fil


def span_search(_dict, len_filters, sgn=None, metric=None):
    if not sgn:
        prepared_data = span.span_search(_dict, len_filters, sgn=None, mode=metric)
        explanations = t.verbalize_field_span_search(prepared_data, _dict)
    else:
        prepared_data = span.span_search(_dict, len_filters, sgn=sgn, mode=metric)
        explanations = t.verbalize_field_span_search(prepared_data, _dict, sgn=sgn)

    return explanations, prepared_data


def convolution_search(_dict, len_filters, sgn=None, metric=None):
    if not sgn:
        prepared_data = fil.convolution_search(_dict, len_filters, sgn=None, mode=metric)
        explanations = t.verbalize_field_span_search(prepared_data, _dict)
    else:
        prepared_data = fil.convolution_search(_dict, len_filters, sgn=sgn, mode=metric)
        explanations = t.verbalize_field_span_search(prepared_data, _dict, sgn=sgn)

    return explanations, prepared_data


def compare_search(previous_searches, samples):
    coincedences = t.compare_search(previous_searches, samples)
    return coincedences


def concatenation_search(self, previous_searches, samples):
    v = t.concatenation_search(previous_searches, samples)
    return v