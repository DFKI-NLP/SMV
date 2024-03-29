

class Source:
    def __init__(self,
                 modelname="Bert",
                 datasetname="IMDB",
                 explainername="Layer Integrated Gradients") -> None:
        """
        on-the-fly implementation of the thermostat library access
        :param modelname: name of the model to be explained
        :param datasetname: name of the dataset to be explained
        :param explainername: name of the explanation generator
        """
        self.model = modelname
        self.dataset = datasetname
        self.explainer = explainername
        self.models = ["albert", "bert", "electra", "roberta", "xlnet"]
        self.datasets = {
            "imdb": "imdb",
            "xnli": "xnli",
            "agnews": "ag_news",
            "ag news": "ag_news",
            "ag_news": "ag_news",
            "multinli": "multi_nli"
        }
        self.explainers = {
            "layer integrated gradients": "lig",
            "occlusion": "occ",
            "lime": "lime",
            "shapley value sampling": "svs",
            "layer deepliftshap": "lds",
            "layer gradientshap": "lgs",
            "layer gradient x activation": "lgxa",
        }
        try:
            self.sourcename = "thermostat/" + \
                              self.datasets[datasetname.lower()] + "-" + modelname.lower() + "-" + self.explainers[explainername.lower()]

            if modelname.lower() == "xlnet" or modelname.lower() == "electra":
                if self.datasets[datasetname.lower()] == "ag_news":
                    print("XLNet and ELECTRA are not supported for ag_news by thermostat. Exiting")
                    raise RuntimeError
        except Exception as e:

            if modelname.lower() not in self.models:
                RuntimeError(f"model didnt match, possible models: {self.models}, entered model: {modelname.lower()}")
            if datasetname.lower() not in list(self.datasets.keys()):
                RuntimeError(f"dataset wasnt valid, possible datasets: {self.datasets}, entered dataset: {datasetname.lower()}")
            if explainername not in list(self.explainers.keys()):
                RuntimeError(
                    f"explainer wasnt valid, possible explainers: {list(self.explainers.keys())}, entered explainer {explainername.lower()}")



    def get_sourcename(self) -> str:
        return self.sourcename

    def get_possible_configurations(self):
        ret = {
            "models": self.models,
            "datasets": self.datasets,
            "explainers": self.explainers
        }
        return ret


class Config:
    def __init__(self,
                 src: Source,
                 sgn: str = "+",
                 samples: int = -1,
                 metric: str = "mean",
                 value: float = 0.4,
                 multiprocessing: bool = False,
                 dev: bool = False,
                 maxwords: int = 100,
                 mincoverage: float = .15):
        """
        Small implementation of a on-the-fly config for rapid testing of multiple settings

        :param src: needs a Source object
        :param sgn: determines which sign to use
        :param samples: determines how many samples to analize
        :param metric: determines which metric to use
        :param value: determines hyperparameter for metric
        :param multiprocessing: determines wether to use multiprocessing
        :param dev: determines wether to use enhanced search parameters
        :param maxwords: if devmode determines wether to look for samples with len < maxwords
        :param mincoverage: if devmode determines wether to look for verbalizations with snippet of coverage > mincoverage
        """

        self.src = src
        self.sgn = sgn
        self.samples = samples
        self.metric = metric
        self.value = value
        self.multiprocessing = multiprocessing
        self.dev = dev
        self.maxwords = dev * maxwords
        self.mincoverage = dev * mincoverage

    def to_yaml(self) -> str:
        yaml_repr = f"source: {self.src.get_sourcename()}\n" \
                    f"sgn: {self.sgn}\n" \
                    f"samples: {self.samples}\n" \
                    f"metric:\n" \
                    f"   name: {self.metric}\n" \
                    f"   value: {self.value}\n" \
                    f"multiprocessing: {self.multiprocessing}\n" \
                    f"dev: {self.dev}\n" \
                    f"maxwords: {self.maxwords}\n" \
                    f"mincoverage: {self.mincoverage}\n"
        return yaml_repr

    def __call__(self, *args, **kwargs):
        return self.to_yaml()
