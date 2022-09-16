import multiprocessing

import src.search_methods.searches as f
import src.search_methods.tools as t

import src.processing.shared_methods as sm


def shared_memory_convsearch(sgn: str,
                             sample_array: dict,
                             len_filters: int,
                             metric,
                             child_pipe) -> None:  # rename
    if not sgn:
        (search, orders) = sm.convolution_search(
            sample_array, len_filters, metric=metric)
    else:
        (search, orders) = sm.convolution_search(
            sample_array, len_filters, sgn, metric=metric)

    child_pipe.send(search, orders)
    child_pipe.close()


def shared_memory_spansearch(sgn: str,
                             sample_array: dict,
                             len_filters: int,
                             metric,
                             child_pipe) -> None:  # rename
    if not sgn:
        (search, orders) = sm.span_search(
            sample_array, len_filters, metric=metric)
    else:
        (search, orders) = sm.span_search(
            sample_array, len_filters, sgn, metric=metric)

    child_pipe.send(search, orders)
    child_pipe.close()


def shared_memory_compare_search(shared_explanations, shared_orders, sample_array):
    shared_explanations["compare search"] = sm.compare_search(shared_orders, sample_array)


def shared_memory_total_search(shared_explanations, sample_array):
    shared_explanations["total order"] = t.verbalize_total_order(t.total_order(sample_array))


def shared_memory_compare_searches(shared_explanations, shared_orders, sample_array):
    shared_explanations["compare searches"] = t.concatenation_search(shared_orders, sample_array)


def check_processes(processes):
    states = []
    for i in processes:
        if i.is_alive():
            states.append(True)
        else:
            states.append(False)
    return states


def check_all_true(ls):
    res = True
    for i in ls:
        if not i:
            res = False
    return res