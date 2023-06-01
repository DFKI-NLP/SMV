import pandas as pd
from typing import Dict

import src.dataloader as data
import src.fastcfg as cfg
import src.tools as tools
import re

IMDB_LABELS = ["positive sentiment", "negative sentiment"]
AGNEWS_LABELS = ["Sci/Tech", "World", "Business", "Sports"]
EXPLANATION_TYPE = "GPT"


def get_dset(dsetname: str) -> Dict[int, list]:
    """
    extracts dataset & attributions
    :param dsetname:
    :return:
    """
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


def get_csv(filepath: str, input_text_col: str = "prompt", explanation_col: str = "completion"):
    """
    extracts csv file
    :param filepath:
    :param input_text_col:
    :param explanation_col:
    :return:
    """
    data = pd.read_csv(filepath)
    results_gpt = {"id": [],
                   "text": [],
                   "gpt_explanation": [],
                   "words": []
                   }

    def append_results():
        results_gpt["id"].append(data["id"][i])
        results_gpt["text"].append(data[input_text_col][i])
        results_gpt["gpt_explanation"].append(data[explanation_col][i])

    for i in range(len(data["id"])):
        if EXPLANATION_TYPE == "GPT":
            append_results()
        elif EXPLANATION_TYPE == "SMV":
            if data["Type"][i] == EXPLANATION_TYPE:
                append_results()

    return results_gpt


def get_citings(text):
    explanation = text.replace("'{placeholder}'", "").replace("{placeholder}", "")
    explanation = explanation.replace("\n", "")
    if EXPLANATION_TYPE == "SMV":
        search_results = [match.group(1)
                          for match in re.finditer(r"»([^«»]+)«", explanation)]
    else:
        search_results = [match.group(1) or match.group(2)
                          for match in re.finditer(r"['\"]([^'\"]+)['\"]", explanation)]

    if "unctuation" in explanation:
        search_results += [".", ",", ":", ";", "-", "!", "?"]

    return search_results
