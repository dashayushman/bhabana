from bhabana.processing import *
from bhabana.datasets import Dataset


class SentimentTreebank(Dataset):

    def __init__(self, mode="regression",
                 use_spacy_vocab=True, load_spacy_vectors=False,
                 spacy_model_name=None, aux=[], cuda=True, rescale=None):
        super(SentimentTreebank, self).__init__("stanford_sentiment_treebank",
                lang="en", size="medium",
                   mode=mode, use_spacy_vocab=use_spacy_vocab,
                       load_spacy_vectors=load_spacy_vectors,
                            spacy_model_name=spacy_model_name,
                                   aux=aux, cuda=cuda, rescale=rescale)
        self.fields = self._load_fields()
        self.line_processor = JSONLineProcessor(self.fields)
        super(SentimentTreebank, self).set_provides(self.fields)
        super(SentimentTreebank, self).set_fields(self.fields)
        super(SentimentTreebank, self).initialize_splits(self.line_processor)

    def _load_fields(self):
        return [
            {
                "key": "text", "dtype": str, "type": "sequence",
                "processors":self._get_text_processing_pipelines()
            },
            {
                "key": "sentiment", "dtype": int, "type": "label",
                "processors": DataProcessingPipeline(self._get_pipeline(
                        type="regression"), name="sentiment",
                        add_to_output=True)
            },
            {
                "key": "label", "dtype": str, "type": "onehot_label",
                "processors": DataProcessingPipeline(self._get_pipeline(
                        type="classification"), name="label",
                        add_to_output=True)
            }
        ]

    def _get_pipeline(self, type):
        if type in self.supported_aux:
            pipeline = [Tokenizer(lang=self.lang,
                                   spacy_model_name=self.spacy_model_name,
                                   mode=type),
                        Seq2Id(self.vocab[type][0], mode=type)]
        elif type == "regression":
            pipeline = []
            if self.rescale:
                pipeline.append(Rescale(1, 10, self.rescale[0], self.rescale[
                    1]))
        elif type == "classification":
            pipeline = [Class2Id(c2i=self.c2i),
                        OneHot(n_classes=self.n_classes)]
        else:
            raise NotImplementedError("{} pipeline has not been implemented "
                                      "yet or it is an invalid pipeline type "
                                      "for the IMDB Dataset".format(type))
        return pipeline

    def _get_text_processing_pipelines(self):
        text_processing_pipeline = []
        for aux in self.aux:
            text_processing_pipeline.append(DataProcessingPipeline(
                    pipeline=self._get_pipeline(aux),
                    name= "text" if aux == "word" else aux,
                    add_to_output=True,
                    add_length=True,
                    add_position=True
            ))
        return text_processing_pipeline