import dataloader

if __name__ == "__main__":
    loader = dataloader.Verbalizer("data/example_xlnet.jsonl", standard_samples=1)
    explanations, texts = loader()
    for Tkey in texts.keys():
        print("Text:")
        text = [word for word in texts[Tkey]["input_ids"]]
        print(*text)
        print()
        for eKey in explanations:
            print("type '{}' explanation:".format(eKey))
            print(*explanations[eKey])
