import datasets
import os
import thermostat
import yaml

import dataloader


if __name__ == "__main__":
    with open("configs/quantile_dev.yml") as stream:
        config = yaml.safe_load(stream)

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

    loader = dataloader.Verbalizer(source, config=config)
    explanations, texts, orders = loader()
    valid_keys = loader.filter_verbalizations(explanations, texts, orders, maxwords=80, mincoverage=.15)
    for key in texts.keys():
        if key in valid_keys:
            txt = "\nSAMPLE:\n" + " ".join(texts[key]["input_ids"])
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
