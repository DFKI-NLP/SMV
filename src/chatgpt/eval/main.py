import json
import tools
from itertools import chain


BASELINE = False
max_k_attribs = 10
cov_per_id = {"imdb": {k: {} for k in range(1, max_k_attribs+1)},
              "agnews": {k: {} for k in range(1, max_k_attribs+1)}
              }
token_rank_cov = {"imdb": {k: 0 for k in range(1, max_k_attribs+1)},
                  "agnews": {k: 0 for k in range(1, max_k_attribs+1)}
                  }
glob_id = 0

def find_score(dset_instance: dict, gpt_words: list):
    global glob_id
    # FIXME: Hack: dont do it like this
    sorted_attrs = [(i[0].lower(), round(i[1], 2))
                    for i in zip(dset_instance["input_ids"], dset_instance["attributions"])
                    if i[1] > 0
                    ]  # ^ Only consider positive attributions
    sorted_attrs.sort(key=lambda x: x[1], reverse=True)

    for k in range(1, max_k_attribs+1):
        k_attribs = sorted_attrs[:k]
        for j, (word, item) in enumerate(k_attribs):
            if not BASELINE:
                if word.lower() in gpt_words:
                    cov_per_id[cur_dataset][k][int(glob_id)] += item
                    if k == max_k_attribs:
                        token_rank_cov[cur_dataset][j+1] += 1
            else:
                cov_per_id[cur_dataset][k][int(glob_id)] += item
                if k == max_k_attribs:
                    token_rank_cov[cur_dataset][j + 1] += 1

        # Coverage @ k for this instance
        try:
            cov_per_id[cur_dataset][k][glob_id] /= sum([i[1] for i in sorted_attrs])
        except ZeroDivisionError:
            print(f"{sum([i[1] for i in k_attribs])} for id {glob_id}")
            cov_per_id[cur_dataset][k][glob_id] = 0


def walk_dataset(dset: dict, gpt_texts: dict):
    global glob_id
    for idx in range(len(gpt_texts["id"])):
        iid = gpt_texts["id"][idx]
        glob_id = iid # instance id for use in coverage file
        for k in range(1, max_k_attribs+1):
            cov_per_id[cur_dataset][k][int(glob_id)] = 0
        text = gpt_texts["gpt_explanation"][idx]
        words = list(chain(*[word.lower().split() for word in tools.get_citings(text)])) # extract all single words
        # and put them in a list
        find_score(dset[iid], words)

    for k in range(1, max_k_attribs+1):
        print(sum(cov_per_id[cur_dataset][k].values())/len(cov_per_id[cur_dataset][k]))


if __name__ == "__main__":
    if tools.EXPLANATION_TYPE == "SMV":
        imdb_csv = "../../data/SMV_IMDb.csv"
        agnews_csv = "../../data/SMV_AGNews.csv"
        input_text_col = "Input Text"
        explanation_col = "Explanation"
    elif tools.EXPLANATION_TYPE == "GPT":
        imdb_csv = "../data/IMDb_ChatGPT_Verbalizations.csv"
        agnews_csv = "../data/AGNews_ChatGPT_Verbalizations.csv"
        input_text_col = "prompt"
        explanation_col = "completion"
    else:
        raise ValueError

    cur_dataset = "imdb"
    dataIMDB = tools.get_dset(cur_dataset)
    csvIMDB = tools.get_csv(imdb_csv, input_text_col=input_text_col, explanation_col=explanation_col)
    walk_dataset(dataIMDB, csvIMDB)
    with open(f"imdbcov_{tools.EXPLANATION_TYPE}.json", "w") as f:
        json.dump(cov_per_id, f) # write down coverage

    cur_dataset = "agnews"
    dataAGNEWS = tools.get_dset(cur_dataset)
    csvAGNEWS = tools.get_csv(agnews_csv, input_text_col=input_text_col, explanation_col=explanation_col)
    walk_dataset(dataAGNEWS, csvAGNEWS)
    with open(f"agnewscov_{tools.EXPLANATION_TYPE}.json", "w") as f:
        json.dump(cov_per_id, f)