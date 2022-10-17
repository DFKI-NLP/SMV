import src.tools as t
import json

from src import dataloader
from vis import Color, color_str


def explain_nodev(config, source, to_json=False):

    key_verbalization_attribs = {}

    modelname = config["source"].replace("thermostat/", "")

    loader = dataloader.Verbalizer(source, config=config)
    explanations, texts, orders = loader()
    returnstr = []

    for key in texts.keys():
        cutoff_top_k_single = 5
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


def explain(cfg_or_path, to_json=False, maxwords=None, mincoverage=None):
    config, source = t.read_config(cfg_or_path)
    if config["dev"]:
        key_verbalization_attribs = {}

        modelname = config["source"].replace("thermostat/", "")

        loader = dataloader.Verbalizer(source, config=config)
        explanations, texts, orders = loader()
        if not maxwords:
            maxwords = loader.maxwords
        if not mincoverage:
            mincoverage = loader.mincoverage
        valid_keys = loader.filter_verbalizations(explanations, texts, orders,
                                                  maxwords=maxwords, mincoverage=mincoverage)
        returnstr = []  # only needed if not to_json

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
        return explain_nodev(config, source, to_json)


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
    sample_text = txt_
    txt = ""
    for expl_subclass in explanations.keys():
        txt += "\nsubclass '{}'".format(expl_subclass)
        if expl_subclass == "compare searches":
            _ = [explanations[expl_subclass][key]]
        else:
            _ = explanations[expl_subclass][key][:cutoff_top_k_single]
        for __ in _:
            if __:
                txt += "\n" + __
            else:
                txt += "\n"
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