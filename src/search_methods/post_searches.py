import warnings
from collections import defaultdict
import numpy as np
from numba import jit


def single_concat_search(sample):
    sample_atts = sample["attributions"]
    input_ids = sample["input_ids"]
    candidates = defaultdict(dict)

    # unused?
    # for stype in searches.keys():
    #     explore_search(candidates, stype, searches, sample_key, sample_atts)



    for i, attr in enumerate(sample_atts):
        candidates['total search'][str(i)] = coverage([i], sample_atts)

    conv_top5 = sorted(candidates['convolution search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]
    span_top5 = sorted(candidates['span search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]
    total_top5 = sorted(candidates['total search'].items(), key=lambda k_v: k_v[1], reverse=True)[:5]

    combined_candidate_indices = []

    combine_results(conv_top5, combined_candidate_indices)
    combine_results(span_top5, combined_candidate_indices)
    combine_results(total_top5, combined_candidate_indices)

    final_spans = []
    for i in sorted(combined_candidate_indices):
        if len(final_spans) > 0 and final_spans[-1][-1] + 1 == i:
            final_spans[-1].append(i)
        else:
            final_spans.append([i])

    cov_fs = []
    for fs in final_spans:
        cov_fs.append(coverage(fs, sample_atts))
    upper_quartile = np.quantile(cov_fs, 0.75)

    num_uq_spans = len([cov_fs[i] > upper_quartile for i, fs in enumerate(final_spans)])
    spans_with_ranks = {}
    for i, fs in enumerate(final_spans):
        rank = sorted(cov_fs, reverse=True).index(cov_fs[i])

        if cov_fs[i] < upper_quartile:
            if num_uq_spans == 0 and rank == 0:
                pass
            else:
                continue

        if len(fs) == 1:
            token = input_ids[fs[0]].replace('Ġ', '')
            verbalization = f"The word » {token} «"
        else:
            span = " ".join([input_ids[t] for t in fs]).replace('Ġ', ' ').replace(
                '<s>', ' ').replace('</s>', '').replace('<pad>', ' ')
            if "." in span:
                verbalization = f"The span » {span} «"
            else:
                verbalization = f"The phrase » {span} «"

        if rank == 0:
            verbalization += " is most important for the prediction"
        else:
            verbalization += " is also salient"

        cov_str = str(round(100 * cov_fs[i]))
        if cov_str == "0":
            continue
        verbalization += " (" + cov_str + " %)."

        spans_with_ranks[rank] = verbalization

    if len(spans_with_ranks) == 0:
        return
    ranked_spans = sorted(spans_with_ranks.items(), key=lambda k_v: k_v[0])

    # TODO: The span.replace(...) has to be different for other models/tokenizers
    # BERT
    # verbalized_explanations[sample_key] = " ".join([span.replace(' ##', '') for i, span in ranked_spans])

    # RoBERTa
    verbalized_explanation = " ".join([span for i, span in ranked_spans])
    # sample_info.append(samples[sample_key])

    return verbalized_explanation


def concatenation_search(samples, *args, **kwargs):
    """
    concatenates sample attributions to largest possible positive attributed span
    :param samples: sample array
    :return: explanation
    """

    # search_types = searches.keys() ##UNUSED##

    # sample_info = []  ##UNUSED##
    verbalized_explanations = {}
    for sample_key in samples.keys():
        verbalized_explanations[sample_key] = single_concat_search(samples[sample_key])
        # sample_info.append(samples[sample_key])
    return verbalized_explanations


def explore_search(candidates, search_type, searches, sample_key, sample_atts):
    warnings.warn("Method not in use")
    candidates[search_type] = {}
    for indices in searches[search_type][sample_key]["indices"]:
        candidates[search_type][','.join([str(idx) for idx in indices])] = coverage(indices, sample_atts)
    return candidates


def coverage(span, attributions):
    if span[0]:
        pos_att_sum = sum([float(a) if a > 0 else 0 for a in attributions])
        if pos_att_sum > 0:
            return sum([attributions[w] for w in span]) / pos_att_sum
    return 0


def combine_results(result_dict, combined_candidate_indices):
    for idx_cov_tuple in result_dict:
        if ',' in idx_cov_tuple[0]:
            indices = idx_cov_tuple[0].split(',')
        else:
            indices = [idx_cov_tuple[0]]
        for idx in indices:
            if idx == 'None':
                continue
            if int(idx) not in combined_candidate_indices:
                combined_candidate_indices.append(int(idx))

