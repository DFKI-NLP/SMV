import pandas as pd
import re
import src.search_methods.fastexplain as fe

data = pd.read_csv("data/agnews.csv")
results_gpt = {"id": [],
               "text": [],
               "gpt_explanation": [],
               "words": []
               }
false_ids = []
false_words = {}

words_inside = 0
words_outside = 0
words_overall = 0

for i in range(len(data["id"])):
    #if data["Type"][i] == "GPT":
    results_gpt["id"].append(data["id"][i])
    results_gpt["text"].append(data["prompt"][i])
    results_gpt["gpt_explanation"].append(data["completion"][i])

for entry in range(len(results_gpt["id"])):
    explanation = results_gpt["gpt_explanation"][entry].replace("'{placeholder}'", "")
    search_results = re.findall(r'"([^"]+)"|\'([^\']+)\'', explanation)
    for text in search_results:
        words_overall += 1
        if text[0].replace("'", "").replace('"', "") in results_gpt["text"][entry]:
            words_inside += 1
        else:
            false_words[results_gpt["id"][entry]] = []
            false_words[results_gpt["id"][entry]].append(text[0].replace("'", "").replace('"', ""))
            words_outside += 1

"""
for i in false_words.keys():
    print("ID:", i)
    for word in false_words[i]:
        print(word)
        false_false = input("is inside of text?")
        if false_false.lower() == "n":
            pass
        else:
            words_inside += 1
            words_outside -= 1
"""
print(len(results_gpt["id"]), "num instances")
print(len(false_words.keys()), "num faulty instances")
print(words_inside, "correctly recited words")
print(words_outside, "wrongly recited words")
print(words_overall, "all recited words/phrases")
print(words_inside/words_overall, "precision")