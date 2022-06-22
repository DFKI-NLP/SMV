# Saliency Map Verbalizations

Coming soon.

---

## Reproduce Greedy Rationales on Annotated LAMBADA (Vafa et al., 2021)

1) Clone [github.com/keyonvafa/sequential-rationales](https://github.com/keyonvafa/sequential-rationales) and change directory (`cd`) to `sequential-rationales`
2) `pip install -r requirements`
3) In *huggingface/rationalize_annotated_lambada.py* (l. 61), replace `model` with pre-trained GPT-2:  
```python
#model = AutoModelForCausalLM.from_pretrained(
#  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
```
4) Change directory (`cd`) to `huggingface` and execute `python rationalize_annotated_lambada.py` producing Greedy Rationales in `huggingface/rationalization_results`