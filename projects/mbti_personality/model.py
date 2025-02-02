# %%
import torch
from transformers import AutoModelForSequenceClassification


class LanguageModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(LanguageModel, self).__init__()
        self.checkpoint = checkpoint

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        match self.checkpoint:
            case "bert-base-uncased":
                for param in self.base_model.bert.parameters():
                    param.requires_grad = False
            case _:
                pass

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs.logits = self.softmax(outputs.logits)

        return outputs
