from utils import inference


class Attack:
    def __init__(self, attack_args):
        self.config = attack_args.config
        self.logger = attack_args.logger
        self.model_word_to_idx = attack_args.model_word_to_idx
        self.spacy_tokenizer = attack_args.data_tokenizer
        self.unk_token = self.config.unk_token

    def inference(self, inputs, model, class_idx, bert_wrapper=None):
        assert isinstance(inputs, str)

        preds, probs = inference(
            inputs,
            model,
            self.model_word_to_idx,
            self.config,
            bert_wrapper=bert_wrapper,
            tokenizer=self.spacy_tokenizer,
            single=True,
        )

        return preds, probs[class_idx]

    def inference_batch(self, inputs, model, class_idx, bert_wrapper=None):
        preds, probs = inference(
            inputs,
            model,
            self.model_word_to_idx,
            self.config,
            bert_wrapper=bert_wrapper,
            tokenizer=self.spacy_tokenizer,
        )

        return preds, [x[class_idx] for x in probs]
