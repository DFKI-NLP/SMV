import src.dataloader as data
import src.search_methods.fastexplain as fe
import src.fastcfg as cfg
import src.tools as tools

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

    source = cfg.Source(modelname="Bert", datasetname="AGNEWS", explainername="LIME")
    config = cfg.Config(src=source,
                        sgn="+",
                        metric="quantile",
                        value=2,
                        multiprocessing=False)
    # It is also possible to use the Verbalizer directly like discussed in readme
    config, source = tools.read_config(config)
    verbalizer = data.Verbalizer(source, config=config)
    explanations, texts, searches = verbalizer()  # aka verbalizer.doit()
    for search_type in explanations:
        for explanation_key in explanations[search_type]:
            print(explanations[search_type][explanation_key])
