import os
import thermostat

import dataloader


if __name__ == "__main__":
    config = {
        "source": "data/Thermostat_imdb-albert-LayerIntegratedGradients.jsonl",
        "sgn": "+",  # TODO: "-"
        "samples": 10,
        "metric": "mean: 0.4",  # TODO: {"name": "mean", "params": .2},
        # searches = {"span", "total"}; "all"
    }

    source = config["source"]
    if os.path.isfile(source):
        loader = dataloader.Verbalizer(source, config=config)
    else:
        df = thermostat.load("imdb-bert-lig")
        # TODO
        raise NotImplementedError

    explanations, texts = loader()

    for key in texts.keys():
        txt = ""
        for word in texts[key]["input_ids"]:
            txt += word.replace("▁", " ")
        print(txt)

        for expl_subclass in explanations.keys():
            print("subclass '{}'".format(expl_subclass))
            _ = explanations[expl_subclass][key][:5]
            for __ in _:
                print(__)

    # TODO: pruned span search?
