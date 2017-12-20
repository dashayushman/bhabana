from bhabana.processing import TextLineProcessor


class TSVLineProcessor(TextLineProcessor):

    def __init__(self, fields):
        super(TSVLineProcessor, self).__init__(fields=fields,
                                               separator="\t")