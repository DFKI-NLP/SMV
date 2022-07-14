import os
import thermostat

import dataloader


if __name__ == "__main__":
    config = {
        "source": "imdb-bert-lig",
        #"source": "data/Thermostat_imdb-albert-LayerIntegratedGradients.jsonl",
        "sgn": "+",  # TODO: "-"
        "samples": 100,
        "metric": "mean: 0.4",  # TODO: {"name": "mean", "params": .2},
        "dev": False
        # searches = {"span", "total"}; "all"
    }

    if not os.path.isfile(config["source"]):
        # Load source from Thermostat configuration
        thermo_config = thermostat.load(config["source"], cache_dir='data')
        # Convert to pandas DataFrame and then to JSON lines
        source = thermo_config.to_pandas().to_json(orient='records', lines=True).splitlines()
    else:
        source = config["source"]
    #TODO: Change to NamedTemporaryFile, this is bad
    #Also TODO: put into module

    loader = dataloader.Verbalizer(source, config=config)
    explanations, texts, _ = loader()
    for key in texts.keys():
        txt = "SAMPLE:\n" + " ".join(texts[key]["input_ids"])
        c = 0
        txt_ = ""
        for i in txt:
            c += 1
            txt_ += i
            if c > 150:
                if i == " ":
                    txt_ += "\n"
                    c = 0
                else:
                    pass
        print(txt_)  # makeshift \n-ing
        for expl_subclass in explanations.keys():
            print("subclass '{}'".format(expl_subclass))
            _ = explanations[expl_subclass][key][:5]
            for __ in _:
                print(__)

        txt = ""

    # TODO: pruned span search?
