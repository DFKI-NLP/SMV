import src.search_methods.fastexplain as fe


if __name__ == "__main__":
    explanations = fe.explain("configs/toy_dev.yml")
    for i in explanations:
        print(i)
