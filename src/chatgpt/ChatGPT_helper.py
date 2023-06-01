import thermostat
import warnings
import re


def thermostat_explain(instance) -> str:
    explan = instance.explanation
    ls: list[str] = []
    for prop in explan:
        round_prop: float = float('{:.2f}'.format(prop[1]))
        ls.append(f"{prop[0]} ({round_prop})")
    return " ".join(ls)


def thermostat_wrapper(model="imdb-bert-lig", sample_keys: list = None) -> list[tuple, ...]:
    instance = thermostat.load(model)
    ls: list[tuple, ...] = []
    for i in sample_keys:
        sample = instance[i]
        label = sample.predicted_label
        ls.append((thermostat_explain(instance=sample), label))
    return ls


def process_data(sample: list = None, importance_scores: list = None) -> str:
    """
    generates sample for zero shot
    """
    if importance_scores:
        if len(sample) != len(importance_scores):
            warnings.warn(f"sample {len(sample)} and scores {len(importance_scores)} unequal .")
        sw_con = zip(sample, importance_scores)
        ls = []
        for conc in sw_con:
            ls.append(f"{conc[0]} ({conc[1]})")
        return " ".join(ls)
    else:
        return " ".join(sample)


def task_generator(task: str = None, sample: list = None, importance_scores: list[float,] = None) -> str:
    std_task = "Classify the following input text into one of the following three categories: [positive, negative, neutral]"
    process_prompt = process_data(sample, importance_scores)
    if task:
        return str(task + ": " + process_prompt)
    else:
        return str(std_task + ": " + process_prompt)


def number_from_str(string: str) -> str:
    """
    string: string containing numbers[float,int,..]
    return: first numbers of string, type(str)
    """
    pattern = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    re.search(pattern=pattern, string=string)
    for catch in re.finditer(pattern, string):
        return catch[0]


def extract_scores(scores: list[str]) -> list:
    """
    scores: list containing output from verbalizer.
    return: list containing importance scores
    """
    ls: list[float] = []
    for string in scores:
        if isinstance(string, str):
            ls.append(float(number_from_str(string)))
        else:
            ls.append(string)
    return ls
