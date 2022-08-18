import src.search_methods.fastexplain as fe


if __name__ == "__main__":
    explanations = fe.explain("configs/quantile_dev.yml")
    for i in explanations:
        print(i)

    # TODO: pruned span search?

    # TODO: pruned span search?
