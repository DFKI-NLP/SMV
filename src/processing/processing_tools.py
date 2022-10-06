import multiprocessing
from typing import List, Tuple

import src.search_methods.searches as f
import src.tools as t
import src.search_methods.post_searches as ps

import src.processing.shared_methods as sm


def worker_convsearch(sgn: str,
                      len_filters: int,
                      metric,
                      child_pipe) -> None:  # rename

    sorted_filters = f.generate_filters(len_filters)
    while True:
        if child_pipe.poll(1):
            key, data = child_pipe.recv()
            if data == -1:
                child_pipe.close()
                break
            coherent_words_sum, coherent_values_sum = f.single_convolution_search(data, sgn, metric, sorted_filters, False)
            _words, _vals = f.result_filtering(coherent_words_sum, coherent_values_sum)
            prepared_data_snippet = {"indices": _words, "values": _vals}
            verbalization = t.single_verbalize_field_span_search(prepared_data_snippet, data, sgn)
            child_pipe.send((prepared_data_snippet, verbalization, key))


def worker_spansearch(sgn: str,
                      len_filters: int,
                      metric,
                      child_pipe) -> None:  # rename

    sorted_filters = f.generate_spans(len_filters)
    while True:
        if child_pipe.poll(.05):
            key, data = child_pipe.recv()
            if data == -1:
                child_pipe.close()
                break
            coherent_words_sum, coherent_values_sum = f.single_convolution_search(data, sgn, metric,
                                                                                  sorted_filters, False)
            _words, _vals = f.result_filtering(coherent_words_sum, coherent_values_sum)
            prepared_data_snippet = {"indices": _words, "values": _vals}
            verbalization = t.single_verbalize_field_span_search(prepared_data_snippet, data, sgn)
            child_pipe.send((prepared_data_snippet, verbalization, key))


def worker_comparesearch(shared_explanations, shared_orders, sample_array):
    shared_explanations["compare search"] = sm.compare_search(shared_orders, sample_array)


def worker_totalsearch(shared_explanations, sample_array):
    shared_explanations["total order"] = t.verbalize_total_order(t.total_order(sample_array))


def worker_concatsearch(sgn: str,
                        len_filters: int,
                        metric,
                        child_pipe) -> None:  # rename
    while True:
        if child_pipe.poll(.05):
            key, data = child_pipe.recv()
            if data == -1:
                child_pipe.close()
                break
            verbalization = ps.single_concat_search(data)
            child_pipe.send((None, [verbalization], key))


def check_processes(processes: List[multiprocessing.Process]) -> List[bool]:
    states = []
    for i in processes:
        if i.is_alive():
            states.append(True)
        else:
            states.append(False)
    return states


def check_all_true(ls) -> bool:
    res = True
    for i in ls:
        if not i:
            res = False
    return res
