import tools

if __name__ == "__main__":
    dataIMDB = tools.get_dset("imdb")
    csvIMDB = tools.get_csv("data/imdb.csv")

    dataAGNEWS = tools.get_dset("agnews")
    csvAGNEWS = tools.get_csv("data/agnews.csv")


