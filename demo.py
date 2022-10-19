import src.search_methods.fastexplain as fe
import src.fastcfg as cfg

if __name__ == "__main__":
    explanations = fe.explain("configs/mean_dev.yml")
    for i in explanations:
        print(i)

    # also this is possible
    config = cfg.Config(cfg.Source())  # for mean_nodev
    explanations = fe.explain(config)
    for i in explanations:
        print(i)

    # or if you want to customize it, you can choose any model, dataset and explainer that is supported by thermostat
    source = cfg.Source(modelname="AlBert", datasetname="AGNEWS", explainername="Shapley Value Sampling")
    config = cfg.Config(src=source,
                        sgn="+",
                        metric="quantile",
                        value=2)
    explanations = fe.explain(config)
    for i in explanations:
        print(i)
