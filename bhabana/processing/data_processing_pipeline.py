from bhabana.processing import Processor


class DataProcessingPipeline:
    def __init__(self, pipeline, name=None, add_to_output=True):

        self.pipeline = self.validate_pipeline(pipeline)
        self.name = self._get_name() if name is None else name
        self.add_to_output = add_to_output

    def _get_name(self):
        name = []
        for processor in self.pipeline:
            name.append(processor.__class__.__name__)
        return "_".join(name)

    @classmethod
    def validate_pipeline(self, pipeline):
        if not hasattr(pipeline, '__iter__'):
            raise Exception("The pipeline must be an iterable. The pipeline "
                            "that you have provided is not an iterable.")
        for processor in pipeline:
            try:
                if not issubclass(processor.__class__, Processor):
                    raise Exception("Your pipeline containes Processors that "
                                    "do not inherit from the Processor Class. "
                                    "Invalid Processor: {}".format(processor.__class__.__name__))
            except:
                raise Exception("Invalid Processor in your pipeline. All "
                                "Processors in a pipeline must be objects of "
                                "classes that inherit from the Processor "
                                "class. Failed while validating object "
                                "of class {}".format(processor))
        return pipeline

    def __call__(self, data):
        return self.run(data)

    def run(self, data):
        prev_output = data
        for processor in self.pipeline:
            validation_status = processor.is_valid_data(data)
            if validation_status["is_valid"]:
                prev_output = processor.process(prev_output)
            else:
                raise Exception("{}".format(validation_status["error"]))
        return prev_output