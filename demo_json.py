import multiprocessing
import os
import src.search_methods.fastexplain as fe



if __name__ == "__main__":
    config_pathes = ["./configs/quantile_dev.yml",]
    jsons = []
    with multiprocessing.Pool(4) as pool:  # assuming you have at least 16 gigs of ram -> assume each process needs 2gigs of ram
        _ = pool.imap(fe.explain_json, config_pathes)

        for entry in _:
            jsons.append(entry)

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
