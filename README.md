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

At the current state of this implementation, you can manipulate the parameters in a config.
The presented example is the "toy_dev.yml".

```python
source: "thermostat/imdb-bert-lig"
sgn: "+"
samples: 100
metric:
  name: "mean"
  value: 0.4
multiprocessing: True
dev: True
maxwords: 100
mincoverage: .1
```
By changing the sgn parameter to "-" or None, you´d allow the verbalizer to take negative values as such, leading to
different results, even though we found "+" to work best in general.
by changing metric to one of our proposed metrics (quantile, mean), you can change the generation of the baseline value
at which a sample snippet gets considered salient and thus returned.
If you enable the dev parameter, you can search for specific classes of samples. Currently implemented is:
the filtering of the length of samples (via maxwords)
and mincoverage, which checks the generated verbalizations for snippets of atleast n% coverage, if no snippet has at
least n% coverage, the sample is not considered valid and thus the index will not be saved.
<br />

This filtering requires some changes to the code from the **Getting started** section
```python
import src.dataloader as d
import src.tools as t

config_path = "configs/toy_dev.yml"
config, source = t.read_config(config_path)
verbalizer = d.Verbalizer(source, config=config, multiprocess=True)

maxwords = 100
mincoverage = 0.1


explanations, texts, searches = verbalizer()
valid_keys = verbalizer.filter_verbalizations(explanations, texts, searches,
                                                  maxwords=maxwords, mincoverage=mincoverage)
for key in valid_keys:
    print(explanations[key])
```

Note that this can also be done via the `src.search_methods.fastexplain.explain` method and a given config,
without the need of changing any code.
Additionally, if youd like to explain a dataset and save the explanations for later use, we´ve implemented a to_json,
that is currently only usable via the `fastexplain.explain` method.

For further information you can look at the documentation of the `Verbalizer` class or our provided demos
