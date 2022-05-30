import dataloader

if __name__ == "__main__":
    loader = dataloader.Verbalizer("data/example_xlnet.jsonl", standard_samples=1)
    explanations, texts = loader()
    print(explanations)
    print(texts)
    for key in explanations.keys():
        q = 0
        for subclass in explanations[key]:
            print("text:")
            print(*[i for i in texts[q]["input_ids"]])
            print("explanation:")
            print(subclass)
            q += 1
