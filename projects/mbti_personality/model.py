# %%
import torch
from transformers import AutoModelForSeq2SeqLM


class LanguageModel(torch.nn.Module):
    def __init__(self, checkpoint):
        super(LanguageModel, self).__init__()
        self.checkpoint = checkpoint

        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
