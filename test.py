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

# pruned span search?
