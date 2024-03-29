import numpy as np


def single_summary(sample: dict, searches: dict[dict], *args, **kwargs) -> str:
    """
    Generates a template-based summary of the most salient tokens based on the candidates from the search methods (Convolutional, Span, Total).
    :param sample: Dictionary representing one instance from the dataset with the following entries:
        Input tokens (`input_ids`)
        Attribution scores (`attributions`)
        Predicted label (`label`)
        Logits (`predictions`)
        Correct prediction boolean (`was_correct`)
    :param searches: Candidate indices from pre-calculated search methods. Each search method is a dictionary of salient
        indices and values (determined by scoring metric like 'mean' or 'quantile')
    :return: Template-based saliency map verbalization
    """
    sample_atts = sample["attributions"]
    input_ids = sample["input_ids"]
    candidates = {}

    """ Explore search (Conv + Span) """
    for stype in searches.keys():  # fixme: please add documentation? - this does the same as explore search
        candidates[stype] = {}
        for indices in searches[stype]["indices"]:
            try:
                candidates[stype][','.join([str(idx) for idx in indices])] = coverage(indices, sample_atts)
            except ZeroDivisionError as e:
                candidates[stype][','.join([str(idx) for idx in indices])] = 0.0
                print("Couldnt calculate coverage, set to zero. This results from sample only having non-positive values.")
            except Exception as e:
                if e != ZeroDivisionError:
                    raise e

    """ Total search """
    candidates["total search"] = {}
    for i, attr in enumerate(sample_atts):
        try:
            candidates["total search"][str(i)] = coverage([i], sample_atts)
        except ZeroDivisionError as e:
            candidates["total search"][str(i)] = 0.0
            print("Couldnt calculate coverage, set to zero. This results from sample only having non-positive values.")
        except Exception as e:
            if e != ZeroDivisionError:
                raise e

    key_retriever = lambda k_v: k_v[1]
    conv_top5 = sorted(candidates["convolution search"].items(), key=key_retriever, reverse=True)[:5]
    span_top5 = sorted(candidates["span search"].items(), key=key_retriever, reverse=True)[:5]
    total_top5 = sorted(candidates["total search"].items(), key=key_retriever, reverse=True)[:5]

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
        try:
            cov_fs.append(coverage((fs[0], fs[-1]), sample_atts))
        except ZeroDivisionError as e:
            cov_fs.append(0.0)
            print("Couldn't calculate coverage, set to zero. This results from sample only having non-positive values.")
        except Exception as e:
            if e != ZeroDivisionError:
                raise e
    upper_quartile = np.quantile(cov_fs, 0.75)

    num_uq_spans = len([cov_fs[i] > upper_quartile for i in range(len(cov_fs))])
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

        """ Coverage percentage not included in this summary
        cov_str = str(round(100 * cov_fs[i]))
        if cov_str == "0":
            continue
        verbalization += " (" + cov_str + " %)."
        """

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


def summarize(samples, searches, *args, **kwargs):
    """
    concatenates sample attributions to largest possible positive attributed span
    :param samples: sample array
    :return: explanation
    """

    # search_types = searches.keys() ##UNUSED##

    # sample_info = []  ##UNUSED##
    verbalized_explanations = {}
    for sample_key in samples.keys():
        search = {}
        for stype in searches.keys():
            search[stype] = searches[stype][sample_key]

        verbalized_explanations[sample_key] = single_summary(samples[sample_key], search)
        # sample_info.append(samples[sample_key])
    return verbalized_explanations


coverage = lambda span, attributions: max(sum(attributions[span[0]:span[-1]])/sum([(a > 0) * a for a in attributions]),
                                          0)
"""
This is the above lambda; just that the lambda is somehow 4s faster
def coverage(span, attributions):
    if span[0]:
        pos_att_sum = [(a > 0) * a for a in attributions]
        pos_att_sum = sum(pos_att_sum)
        if pos_att_sum > 0:
            cov = attributions[span[0]:span[-1]]
            cov = sum(cov)
            cov = cov/pos_att_sum
            return cov
    return 0
"""


def combine_results(search_candidates: list[tuple[str, float]], combined_candidate_indices: list) -> None:
    """
    Appends the results from one search to a joint candidate list of salient spans
    :param search_candidates: Sorted list of candidates (Tuples of strings representing salient indices and coverage
     value) from one search method (Convolutional, Span, Total)
    :param combined_candidate_indices: Joint candidate list of salient spans
    """
    for idx_cov_tuple in search_candidates:
        if ',' in idx_cov_tuple[0]:
            indices = idx_cov_tuple[0].split(',')
        else:
            indices = [idx_cov_tuple[0]]
        for idx in indices:
            if idx == 'None':
                continue
            if int(idx) not in combined_candidate_indices:
                combined_candidate_indices.append(int(idx))

