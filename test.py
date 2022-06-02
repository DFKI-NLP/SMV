import dataloader
# dataset @ "https://cloud.dfki.de/owncloud/index.php/s/zjMddcqewEcwSPG/download", put into data folder


if __name__ == "__main__":

    config = {
        "sgn": "+",
        "samples": 1,
        "metric": "mean: 1"
    }

    loader = dataloader.Verbalizer("data/Thermostat_imdb-albert-LayerIntegratedGradients.jsonl", config=config)
    explanations, texts = loader()
    for Tkey in texts.keys():
        print("Text:")
        text = [word for word in texts[Tkey]["input_ids"]]
        print(*text)
        print()
        for eKey in explanations:
            print("type '{}' explanation:".format(eKey))
            print(*explanations[eKey])
