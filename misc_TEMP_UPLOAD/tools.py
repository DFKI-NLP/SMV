import pandas as pd
import src.dataloader as data
import src.fastcfg as cfg
import src.tools as tools
import re

def get_dset(dsetname: str) -> dict[list]:
    dsetname = dsetname.lower()
    if dsetname == "agnews":
        source = cfg.Source(modelname="Bert", datasetname="AGNEWS", explainername="Layer Integrated Gradients")
        config = cfg.Config(source, "+")
        config, source = tools.read_config(config)
        verbalizer = data.Verbalizer(source, config=config)
        return verbalizer.read_samples()
    if dsetname == "imdb":
        source = cfg.Source(modelname="Bert", datasetname="IMDB", explainername="Layer Integrated Gradients")
        config = cfg.Config(source, "+")
        config, source = tools.read_config(config)
        verbalizer = data.Verbalizer(source, config=config)
        return verbalizer.read_samples()
    raise Exception("Specified dataset doesn't exist.")


def get_csv(filepath: str):
    data = pd.read_csv(filepath)

    _data = {}
    for entry in range(len(data["ID"])):
        key = data["ID"][entry]
        if data["Type"][entry] == "GPT":
            _data[key] = data["Explanation"][entry]
    return _data

def get_citings(text):
    citings = re.search(r'"([^"]*)"' + r"'([^']*)'", text)
