import src.search_methods.fastexplain as fe


if __name__ == "__main__":
    explanations = fe.explain("configs/quantile_dev.yml", to_json=True)
    #print(explanations)
