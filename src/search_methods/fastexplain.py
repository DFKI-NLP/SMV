import datasets
import os
import pandas as pd
import thermostat
import yaml
import json

from src import dataloader
from vis import Color, color_str, save_heatmap_as_image

def explain_nodev(config, to_json=False):
    if not os.path.isfile(config["source"]):
        if "thermostat/" in config["source"]:
            # Load source from Thermostat configuration
            thermo_config_name = config["source"].replace("thermostat/", "")
            thermo_config = thermostat.load(thermo_config_name, cache_dir="data")
            # Convert to pandas DataFrame and then to JSON lines
            source = thermo_config.to_pandas().to_json(orient="records", lines=True).splitlines()
        elif "datasets/" in config["source"]:
            # TODO: Add annotator rationale datasets
            # https://huggingface.co/datasets/movie_rationales
            # http://www.eraserbenchmark.com/ (BoolQ)
            # Load source from Hugging Face datasets library
            hf_dataset_name = config["source"].replace("datasets/", "")
            hf_dataset = datasets.load_dataset(hf_dataset_name, cache_dir="data")
            raise NotImplementedError("HF datasets not supported atm.")
        else:
            raise NotImplementedError("Invalid source!")
    else:
        source = config["source"]
    #TODO: Change to NamedTemporaryFile, this is bad
    #Also TODO: put into module

    key_verbalization_attribs = {}

    modelname = config["source"].replace("thermostat/", "")

    loader = dataloader.Verbalizer(source, config=config)
    explanations, texts, orders = loader()
    if not to_json:
        returnstr = []

    for key in texts.keys():
        cutoff_top_k_single = 5
        txt, sample_text = txt = to_string(explanations, texts, key, cutoff_top_k_single)
        if to_json:
            key_verbalization_attribs[key] = {"sample": texts[key]["input_ids"],
                                              "verbalization": txt,
                                              "attributions": texts[key]["attributions"]}
        else:
            returnstr.append(sample_text + "\n" + txt)

    if to_json:
        key_verbalization_attribs["modelname"] = modelname
        res = json.dumps(key_verbalization_attribs)
        return res
    else:
        return returnstr


def explain(config_path, to_json=False):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    if config["dev"]:
        if not os.path.isfile(config["source"]):
            if "thermostat/" in config["source"]:
                # Load source from Thermostat configuration
                thermo_config_name = config["source"].replace("thermostat/", "")
                thermo_config = thermostat.load(thermo_config_name, cache_dir="data")
                # Convert to pandas DataFrame and then to JSON lines
                source = thermo_config.to_pandas().to_json(orient="records", lines=True).splitlines()
            elif "datasets/" in config["source"]:
                # TODO: Add annotator rationale datasets
                # https://huggingface.co/datasets/movie_rationales
                # http://www.eraserbenchmark.com/ (BoolQ)
                # Load source from Hugging Face datasets library
                hf_dataset_name = config["source"].replace("datasets/", "")
                hf_dataset = datasets.load_dataset(hf_dataset_name, cache_dir="data")
                raise NotImplementedError("HF datasets not supported atm.")
            else:
                raise NotImplementedError("Invalid source!")
        else:
            source = config["source"]
            thermo_config = None
        #TODO: Change to NamedTemporaryFile, this is bad
        #Also TODO: put into module

        key_verbalization_attribs = {}

        modelname = config["source"].replace("thermostat/", "")

        loader = dataloader.Verbalizer(source, config=config)
        explanations, texts, orders = loader()
        valid_keys = loader.filter_verbalizations(explanations, texts, orders, maxwords=80, mincoverage=.15)
        if not to_json:
            returnstr = []

        cutoff_top_k_single = 5
        for key in texts.keys():
            if key in valid_keys:
                key_verbalization_attribs[key] = {}
                txt, sample_text = to_string(explanations, texts, key, cutoff_top_k_single)
                if to_json:
                    key_verbalization_attribs[key] = {"sample": texts[key]["input_ids"],
                                                      "verbalization": txt,
                                                      "attributions": texts[key]["attributions"]}
                else:
                    returnstr.append(sample_text + "\n" + txt)

        if to_json:
            key_verbalization_attribs["modelname"] = modelname
            res = json.dumps(key_verbalization_attribs)
            return res
        else:
            return returnstr
    else:
        return explain_nodev(config, to_json)


def explain_json(config_path):
    return explain(config_path, True)


def to_string(explanations, texts, key, cutoff_top_k_single):
    txt = "\nSAMPLE:\n"
    fmtd_tokens = []
    for i, token in enumerate(texts[key]["input_ids"]):
        if texts[key]["attributions"][i] >= sorted(
                texts[key]["attributions"], reverse=True)[cutoff_top_k_single - 1]:
            fmtd_tokens.append(color_str(color_str(token, Color.RED), Color.BOLD))
        elif texts[key]["attributions"][i] > 0:
            fmtd_tokens.append(color_str(token, Color.BOLD))
        else:
            fmtd_tokens.append(token)
    txt += " ".join(fmtd_tokens)
    sample_text = txt
    c = 0
    txt_ = ""
    for i in txt:
        c += 1
        txt_ += i
        if c > 150:
            if i == " ":
                txt_ += "\n"  # makeshift \n-ing
                c = 0
            else:
                pass
    sample = txt_
    txt = ""
    for expl_subclass in explanations.keys():
        txt += "\nsubclass '{}'".format(expl_subclass)
        _ = explanations[expl_subclass][key][:cutoff_top_k_single]
        for __ in _:
            txt += "\n" + __
    txt += "\nPrediction was correct." if texts[key]["was_correct"] else "\nPredicton was incorrect"
    return txt, sample_text


def heatmap_com_verb(key, valid_keys, thermo_config, thermo_config_name, explanations):
    if key in valid_keys:
        #if thermo_config:
        #    thermounit = thermo_config[key]
        #    """ Only execute the code once! """
        #    save_heatmap_as_image(thermounit.heatmap, filename=f"{thermo_config_name}/{key}.png")

        # sample = " ".join(texts[key]["input_ids"])  ##UNUSED?##
        if key in explanations["compare searches"]:
            smv = explanations["compare searches"][key]

    return smv