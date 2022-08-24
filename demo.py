import src.search_methods.fastexplain as fe


if __name__ == "__main__":
    explanations = fe.explain("configs/mean_nodev.yml")
    for i in explanations:
        print(i)
