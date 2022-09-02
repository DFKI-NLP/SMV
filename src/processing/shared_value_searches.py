import src.search_methods.spans as s
import src.search_methods.filters as f
import src.search_methods.tools as t

import src.processing.shared_methods as sm


def shared_memory_convsearch(sgn, sample_array, len_filters, metric, shared_obj):
    if not sgn:
        (search, orders) = sm.convolution_search(
            sample_array, len_filters, metric=metric)
    else:
        (search, orders) = sm.convolution_search(
            sample_array, len_filters, sgn, metric=metric)

    shared_obj["convolution search"] = (search, orders)


def shared_memory_spansearch(sgn, sample_array, len_filters, metric, shared_obj):
    if not sgn:
        (search, orders) = sm.span_search(
            sample_array, len_filters, metric=metric)
    else:
        (search, orders) = sm.span_search(
            sample_array, len_filters, sgn, metric=metric)

    shared_obj["span search"] = (search, orders)

def shared_memory_compare_search(shared_explanations, shared_orders, sample_array):
    shared_explanations["compare search"] = sm.compare_search(shared_orders, sample_array)


def shared_memory_total_search(shared_explanations, sample_array):
    shared_explanations["total order"] = t.verbalize_total_order(t.total_order(sample_array))

def shared_memory_compare_searches(shared_explanations, shared_orders, sample_array):
    shared_explanations["compare searches"] = t.compare_searches(shared_orders, sample_array)
