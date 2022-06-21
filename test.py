import dataloader
# dataset @ "https://cloud.dfki.de/owncloud/index.php/s/zjMddcqewEcwSPG/download", put into data folder


if __name__ == "__main__":

    config = {
        "sgn": "+",  # Vorzeichenfehler bei "-"
        "samples": 10,
        "metric": "mean: 0.4"# {"name": "mean", "params": .2},
        # searches = {"span", "total"}; "all"
    }

    loader = dataloader.Verbalizer("data/Thermostat_imdb-albert-LayerIntegratedGradients.jsonl", config=config)
    # loader = dataloader.Verbalizer.from_thermostat(config)
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

# pruned span search?


