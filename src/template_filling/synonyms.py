from src.template_filling.utils import TemplateElement


SINGLE_FEATURE_NAME = TemplateElement({
    "word": 0.4,
    "token": 0.3,
    "feature": 0.3,
})

PHRASE_NAME = TemplateElement({
    "span": 0.5,
    "phrase": 0.5,
})

IMPORTANT_NAME = TemplateElement({
    "important": 0.5,
    "salient": 0.3,
    "influential": 0.1,
    "impactful": 0.1,
})

PREDICTION_NAME = TemplateElement({
    "prediction": 0.4,
    "decision": 0.15,
    "classification": 0.1,
    "predicted label": 0.1,
    "outcome": 0.05,
    "model prediction": 0.05,
    "model behavior": 0.025,
    "model\'s behavior": 0.025,
    "model\'s predicted label": 0.025,
    "model\'s prediction": 0.025,
    "model\'s judgment": 0.025,
    "prediction of the classifier": 0.025,
})
