# Saliency Map Verbalization

<div align="center">
    <h3>Verbalizing saliency maps with templates and binary filtering as well as instruction-based LLMs</h3>
</div>

![alt text](imgs/SMV_Fig1_2023-05-04.png)

[![arXiv](https://img.shields.io/badge/arXiv-2210.07222-red)](http://arxiv.org/abs/2210.07222)


## Getting started:
1) You should use Python 3.8
2) clone this repository
3) `pip install -r requirements`

## Example usage
To verbalize a dataset you first need to write a config file, the rest will be managed by the `Verbalizer` class object.
We provide some examplatory config files to play around with.
After defining a config you can use it to immediately get an explanation.
For a fast start, look at our demo.py, if you only want a fast explanation, that is all you need.

```python
from src.search_methods import fastexplain as fe

config_path = "configs/toy_dev.yml"
explanation_string = fe.explain(config_path)
for explanation in explanation_string:
    print(explanation)
```

Just like in demo.py. Output (one explanation).:
```
SAMPLE:
fantastic , madonna at her finest , the film is funny and her 
acting is brilliant . it may have been made in the 80 ' s but it has all the 
qualities of a modern hollywood block - buster . i love this 
film and i think its totally unique and will cheer up any dr ##oop 
##y person within a matter of minutes . fantastic .

subclass 'convolution search'
snippet: 'i love this ' contains 53.51% of prediction score.
snippet: 'love this and ' contains 44.96% of prediction score.
snippet: '. love this ' contains 43.72% of prediction score.
snippet: 'love this i ' contains 43.67% of prediction score.
snippet: 'love this film ' contains 42.52% of prediction score.
subclass 'span search'
snippet: 'i love this film and ' contains 57.03% of prediction score.
snippet: '. i love this film ' contains 55.78% of prediction score.
snippet: 'i love this ' contains 53.51% of prediction score.
snippet: 'love this film and i ' contains 47.19% of prediction score.
snippet: 'love this film ' contains 42.52% of prediction score.
subclass 'concatenation search'
The phrase » i love this « is most important for the prediction (54 %).
subclass 'compare search'
snippet: 'this film and' occurs in all searches and accounts for 28.74% of prediction score
snippet: 'love this film' occurs in all searches and accounts for 42.52% of prediction score
snippet: 'i love this' occurs in all searches and accounts for 53.51% of prediction score
snippet: '. i love' occurs in all searches and accounts for 30.03% of prediction score
subclass 'total order'
top tokens are:
token 'this' with 25.22% of prediction score
token 'love' with 16.76% of prediction score
token 'i' with 11.53% of prediction score
token 'unique' with 3.98% of prediction score
Prediction was correct.
```
Note that the original output will be colourcoded

### How our search methods work

![alt text](imgs/SMV_Temp.jpg)


## Advanced

Otherwise, if you don't like the format in which we represent explanations, you can get the raw output of our search
methods like this, by using the `Verbalizer` directly.
```python
import src.dataloader as data
import src.tools as tools

config_path = "configs/toy_dev.yml"
config, source = tools.read_config(config_path)
verbalizer = data.Verbalizer(source, config=config)

explanations, texts, searches = verbalizer()

for search_type in explanations:
    for explanation_key in explanations[search_type]:
        print(explanations[search_type][explanation_key])
```
Note that `verbalizer()` calls `verbalizer.doit()`
also multiprocess is set to `True` by default, disabling is encouraged for systems **with less than 8GB RAM**
or systems with **less than 4 (physical) cores**. disabling multiprocessing can lead **to 5x increased running time**.

This will produce the same explanation like the demo but the resulting string is not formatted and there will be
less salient findings too.
The variable `texts` will contain the samples of the dataset you chose to explain, `searches` will contain our
calculated values for span- and convolution search (`np.array`).
`explanations` itself will be a dictionary, that is ordered like this:

|          | Top layer                                                                             | Accessed layer             |
|:---------|:--------------------------------------------------------------------------------------|:---------------------------|
| values   | multiple `dict` objects                                                               | `list` of `string`         |
| keys     | "span_search", "convolution_search", "compare search", "total order", "summarization" | `string` like "1", "2",... |

![alt text](imgs/graph_0.jpg)

### Manual config writing
You currently have two methods of generating a config. The first one is manual.
The presented example is the "toy_dev.yml".

```yaml
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

### Config constructor
Our second method of building a config file is a small plug-and-play like system.
```python
import src.fastcfg as cfg
import src.search_methods.fastexplain as fe

# fixme: only lig and occ implemented in converter in src.fastcfg.Source, implement rest too.

source = cfg.Source(modelname="Name of your model, for example Bert",
                    datasetname= "Name of the dataset, for example IMDB",
                    explainername= "Full name of the explanation algorithm, for example Layer Integrated Gradients")
config = cfg.Config(src=source,
                    sgn= "+",
                    samples= 100)
# With Config.get_possible_configurations() you can get a dictionary containing all possible configurations i.e. models,
# datasets and explainers
explanations = fe.explain(config)
for explanation in explanations:
    print(explanation)

# you can also save a generated Config:
filename = "filename.yml"
with open(filename) as f:
    f.write(config.to_yaml())
```
With this you can change specific parameters on-the-fly for fast-testing of multiple configurations.

### Filtering of results

Our filtering methods require some changes to the code from the **Getting started** section.
```python
import src.dataloader as data
import src.tools as tools

config_path = "configs/toy_dev.yml"
config, source = tools.read_config(config_path)
verbalizer = data.Verbalizer(source, config=config, multiprocess=True)

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
Additionally, if you want to explain a dataset and save the explanations for later use, we´ve implemented a to_json,
that is currently usable via the `fastexplain.explain` method, by setting the to_json parameter to `True`.

For further information you can look at the documentation of the `Verbalizer` class or our provided demos
Most of our code is documented and built to be changed easily.

## Search Types
As proposed in our paper, we employ different search methods to search for salient snippets. You can set your desired
searches by changing the ```mode``` parameter of ```dataloader.Verbalizer.doit()```. Default employs all our
algorithms.<br/>

| Name               | Description                                                     |
|:-------------------|:----------------------------------------------------------------|
| convolution search | implements our proposed Convolution Search                      |
| span search        | implements our proposed Span Search                             |
| compare search     | filters for multiple equal results in convolution & span search |
| total order        | filters for top-k tokens                                        |
| summarization      | implements our proposed Summarized Explanation                  |


## Config parameter cheat-sheet

| Parameter         | Values                                    | Description                                                                                                      | Dtype(s)      |
|:------------------|:------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:--------------|
| `source`          | Path to file                              | Path to config file                                                                                              | `str`         |
| `multiprocessing` | `True`, `False`: `True` is default        | Should our multiprocessing implementation of our paper be used                                                   | `bool`        |
| `sgn`             | `"+"`, `"-"`, `None`                      | Values of what sign should be used for calculation, None uses all                                                | `str`, `None` |
| `samples`         | Any of {-1, (0, +oo]}                     | -1 to read whole dataset, any other number to read                                                               | `int`         |
| metric:`name`     | See documentation of dataloader           | How should the baseline value be calculated                                                                      | `str`         |
| metric:`value`    | Depends on metric, see docs of dataloader | What value should be used to generate baseline value                                                             | `float`       |
| `dev`             | `True`, `False`, default is `False`       | Enables further settings, allowing to filter the dataset, if False, `maxwords` and `mincoverage` will be ignored | `bool`        |
| `maxwords`        | any of (0, +oo]                           | Filters for samples that have a maximum of `maxwords` words                                                      | `int`         |
| `mincoverage`     | any of [0., 1.]                           | Filters for samples with a snippet of at least `mincoverage`% of coverage                                        | `float`       |

Please note that this is still in development and object to change


# GPT verbalizations

[Click here](src/chatgpt/README.md)


# Citation

```bibtex
@inproceedings{feldhus-2023-smv,
    title = "Saliency Map Verbalization: Comparing Feature Importance Representations from Model-free and Instruction-based Methods",
    author = "Nils Feldhus and Leonhard Hennig and Maximilian Dustin Nasert and Christopher Ebert and Robert Schwarzenberg and Sebastian M\"{o}ller",
    booktitle = "Proceedings of the First Workshop on Natural Language Reasoning and Structured Explanations (NLRSE)",
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2210.07222",
}
```

ACL Anthology version to be added in July.
