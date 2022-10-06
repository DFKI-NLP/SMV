import multiprocessing
import os
import src.search_methods.fastexplain as fe
import json


if __name__ == "__main__":
    config_pathes = ["./configs/quantile_dev.yml", ]
    jsons = []
    for config_path in config_pathes:
        jsons.append(fe.explain_json(config_path))

    try:
        for i in range(len(config_pathes)):
            f = open("./tmp/temp_{}.json".format(i), mode="w")
            f.write(jsons[i])
            f.close()
    except FileNotFoundError:
        os.mkdir("./tmp/")
        for i in range(len(config_pathes)):
            f = open("./tmp/temp_{}.json".format(i), mode="w")
            f.write(jsons[i])
            f.close()

    f = open("tmp/temp_0.json")
    jsonf = json.load(f)
    f.close()
    for sample_key in jsonf.keys():
        print(jsonf[sample_key]["verbalization"])

