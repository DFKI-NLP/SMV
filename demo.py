import src.search_methods.fastexplain as fe
import src.fastcfg as fc


if __name__ == "__main__":
    explanations = fe.explain("configs/toy_dev.yml")
    for i in explanations:
        print(i)

    # also this is possible
    config = fc.Config(fc.Source())  # for mean_nodev
    explanations = fe.explain(config)
    for i in explanations:
        print(i)

    # or if you want to customize it
    source = fc.Source(modelname="AlBert", datasetname="AGNEWS", explainername="Shapley Value Sampling")
    config = fc.Config(src=source,
                       sgn="+",
                       metric="quantile",
                       value=2)
    explanations = fe.explain(config)
    for i in explanations:
        print(i)
