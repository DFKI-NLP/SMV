import src.processing.process_handlers as ph
import yaml
import os
import thermostat
import datasets
import src.dataloader as dataloader

config_path = "configs/mean_dev.yml"

with open(config_path) as stream:
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
# TODO: Change to NamedTemporaryFile, this is bad
# Also TODO: put into module

key_verbalization_attribs = {}

modelname = config["source"].replace("thermostat/", "")

loader = dataloader.Verbalizer(source, config=config)

tasks = [ph.span_task(), ph.conv_task()]

Handler = ph.ProcessHandler(loader, tasks)