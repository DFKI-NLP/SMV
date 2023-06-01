from tqdm import tqdm

from ChatGPT_helper import thermostat_wrapper
from ChatGPT_class import chatgpt_handler
from SMV_Lists import imdb_SMV, agnews_SMV


OPENAI_MODEL_ID = "gpt-3.5-turbo"


def task_instruction(model_description: str, text_description: str, label: str, objective: str = 'commonsense'):
    intro = f"A {model_description} has predicted this text as '{label}'. " \
        f"The scores behind each word indicate how important it was for the {model_description} " \
        f"to predict '{label}'. " \
        f"The scores have been determined after the {model_description} has already made its prediction. " \
        f"The {model_description} cannot base its prediction on the scores, " \
        f"only on the {text_description} itself. \n"

    if objective == 'commonsense':
        intro += f"Based on the importance scores, briefly explain why the {model_description} " \
            f"predicted this {text_description} as '{label}': "
    elif objective == 'saliency':
        intro += f"In a nutshell, what are the most important words or linguistic units in the {text_description}" \
            f"for the '{label}' prediction? "

    return intro

def thermostat_task_ag_news(sample: str, label: str):
    model_desc = 'topic classifier'
    text_desc = 'news article'
    label_str = label + ' topic'

    instruction = task_instruction(model_desc, text_desc, label_str)
    return f"{text_desc} with importance scores: {sample}. \n{instruction}"


def thermostat_task_imdb(sample, label):
    model_desc = 'sentiment analyzer'
    text_desc = 'movie review'
    label_dict = {'pos': 'positive sentiment',
                  'neg': 'negative sentiment'}
    label_str = label_dict[label]

    instruction = task_instruction(model_desc, text_desc, label_str)
    return f"{text_desc} with importance scores: {sample}. \n{instruction}"


def imdb(key_list: list) -> None:
    imdb_chat = chatgpt_handler(OPENAI_MODEL_ID)
    imdb_chat.use_wandb()
    imdb_samples = thermostat_wrapper(model="imdb-bert-lig", sample_keys=key_list)

    for i in tqdm(range(len(imdb_samples)), total=len(imdb_samples)):
        prompt = thermostat_task_imdb(imdb_samples[i][0], imdb_samples[i][1])
        imdb_chat.chat_request(prompt, id=key_list[i])
    imdb_chat.visualize()
    return


def agnews(key_list: list) -> None:
    ag_news_chat = chatgpt_handler(OPENAI_MODEL_ID)
    ag_news_chat.use_wandb()
    agnews_samples = thermostat_wrapper(model="ag_news-bert-lig", sample_keys=key_list[:5])

    for i in tqdm(range(len(agnews_samples)), total=len(agnews_samples)):
        prompt = thermostat_task_ag_news(agnews_samples[i][0], agnews_samples[i][1])
        ag_news_chat.chat_request(prompt, id=key_list[i])
    ag_news_chat.visualize()
    return


if __name__ == "__main__":
    imdb(imdb_SMV)
    agnews(agnews_SMV)
