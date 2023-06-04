


def build_summary(candidates) -> str:
    """ Builds a verbalization out of options and rules. """
    for rank, cd in enumerate(candidates):
        rank = candidates.index(cov_fs[i])

        if "." in span:
            verbalization = f"The span » {span} «"
        else:
            verbalization = f"The phrase » {span} «"

        if rank == 0:
            verbalization += " is most important for the prediction"
        else:
            verbalization += " is also salient"

    # TODO: Add punctuation at the end of the verbalization

    # TODO: The span.replace(...) has to be different for other models/tokenizers
    # BERT
    # verbalized_explanations[sample_key] = " ".join([span.replace(' ##', '') for i, span in ranked_spans])

    return verbalized_explanation
