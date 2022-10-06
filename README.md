# Saliency Map Verbalizations

This repository implements [link zu unserem Paper]

## Getting started:
1) You should use Python 3.8
2) clone this repository
3) `pip install -r requirements`

To verbalize a dataset you first need to write a config file, the rest will be managed the `Verbalizer`. We provide some
examplatory config files to play around with.
After defining a config you can use it to immediately get an explanation.
```python
from src.search_methods import fastexplain as fe

config_path = "configs/toy_dev.yml"  # TODO: modularize as standard-config/ build dataclass
explanation_string = fe.explain(config_path)
for explanation in explanation_string:
    print(explanation)
```
just like in demo.py

if you'd like to use our explanation methods more in depth you can use the `Verbalizer` directly
```python
import src.dataloader as d
import src.tools as t

config_path = "configs/toy_dev.yml"
config, source = t.read_config(config_path)
verbalizer = d.Verbalizer(source, config=config, multiprocess=True)

explanations, texts, searches = verbalizer()

"""
note that verbalizer() calls verbalizer.doit()
also multiprocess is set to True by default, disabling is encouraged for systems with less than 8GB RAM
or systems with less than 4 (logic) cores

TODO: automate multiprocessing decision making

disabling multiprocessing can lead to 5x increased running time
"""

for explanation in explanations:
    print(explanation)
```
This will produce the same explanation like the demo but the resulting string is not formatted.
The variable `texts` will contain the samples of the dataset you chose to explain, `searches` will contain our
calculated values for span- and convolution search.

# Advanced

---

## Reproduce Greedy Rationales on Annotated LAMBADA (Vafa et al., 2021)

1) Clone [github.com/keyonvafa/sequential-rationales](https://github.com/keyonvafa/sequential-rationales) and change directory (`cd`) to `sequential-rationales`
2) `pip install -r requirements`
3) In *huggingface/rationalize_annotated_lambada.py* (l. 61), replace `model` with pre-trained GPT-2:  
```python
#model = AutoModelForCausalLM.from_pretrained(
#  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
```
4) Change directory (`cd`) to `huggingface` and execute `python rationalize_annotated_lambada.py` producing Greedy Rationales in `huggingface/rationalization_results`