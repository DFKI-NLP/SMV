import dataloader
# dataset @ "https://cloud.dfki.de/owncloud/index.php/s/zjMddcqewEcwSPG/download", put into data folder


if __name__ == "__main__":

    config = {
        "sgn": "+",  # Vorzeichenfehler bei "-"
        "samples": 5,
        "metric": "mean: .4"# {"name": "mean", "params": .2}
    }

    loader = dataloader.Verbalizer("data/Thermostat_imdb-albert-LayerIntegratedGradients.jsonl", config=config)
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
