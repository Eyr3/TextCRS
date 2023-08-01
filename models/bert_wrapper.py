"""
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from utils import print_model_state_dict, list_join


class BertWrapper:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=self.config.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        print_model_state_dict(logger, self.model)

    def pre_pro(self, text):
        
        assert isinstance(text, list)

        tokens = self.tokenizer.encode_plus(
            list_join(text),
            max_length=self.config.bert_max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        return tokens["input_ids"], tokens["attention_mask"]


class bertWrapper:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # self.model = BertForSequenceClassification.from_pretrained(
        #     "bert-base-uncased",
        #     num_labels=self.config.num_classes,
        #     output_attentions=False,
        #     output_hidden_states=False,
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=self.config.bert_max_len,
            do_lower_case=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.config.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        print_model_state_dict(logger, self.model)

    def pre_pro(self, text):
        assert isinstance(text, list)

        tokens = self.tokenizer.encode_plus(
            list_join(text),
            max_length=self.config.bert_max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        return tokens["input_ids"], tokens["attention_mask"]
