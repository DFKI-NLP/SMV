import openai
from pprint import pprint


class chatgpt_handler:
    # Please enter API key and organization here:
    api_key = ""
    organization = ""

    def __init__(self, model: str = None):
        openai.organization = chatgpt_handler.organization
        openai.api_key = chatgpt_handler.api_key
        self.avail_models = []
        self.responses = []
        self.wandb_true = False
        self.model = model

    def use_wandb(self) -> None:
        import wandb
        self.wandb_true = True
        wandb.init(project='SMV')
        self.prediction_table = wandb.Table(columns=["prompt", "id", "completion"])

    def list_models(self) -> list:
        models = openai.Model.list()
        for i in models["data"]:
            self.avail_models.append(i["id"])
        print(self.avail_models)
        return self.avail_models

    def static_request(self, prompt: str = None, temperature: float = 0.5, max_tokens: int = 10, top_p: float = 1.0,
                       stop: str = "\n") -> list:
        prompt = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        print(response)
        return response["choices"][0]["message"]["content"]

    def chat_request(self, prompt: str = None, temperature: float = 0.5, max_tokens: int = 300, top_p: float = 1.0,
                     stop: str = "\n", id=None) -> list:
        if prompt is not None:
            prompt = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        self.responses.append(response)

        if self.wandb_true:
            self.prediction_table.add_data(prompt, id, response["choices"][0]["message"]["content"])
        return response

    def visualize(self) -> None:
        if self.wandb_true:
            import wandb
            wandb.log({"predictions": self.prediction_table})
            wandb.finish()
        else:
            pprint(self.responses)
