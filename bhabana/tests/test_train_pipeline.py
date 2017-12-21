import os
import shutil
import time

from nose.tools import *

import bhabana.utils as utils
from bhabana.trainer.brahmaputra import ex
from bhabana.trainer.brahmaputra import experiment_boilerplate


class TestTrainerScript:

    @classmethod
    def validate_experiment_dir(cls, experiment_config):
        assert_true(experiment_config["experiment_root_dir"] ==
                    utils.EXPERIMENTS_DIR)
        assert_true(
            os.path.exists(experiment_config["experiment_dir"]))
        assert_true(
            os.path.exists(experiment_config["checkpoints_dir"]))
        assert_true(os.path.exists(experiment_config["logs_dir"]))

        shutil.rmtree(experiment_config["experiment_dir"])

    def test_experiment_configs(self):
        r = ex.run()
        self.validate_experiment_dir(r.config["experiment_config"])

    def test_experiment_boilerplate(self):
        pipeline = "something"
        setup = {
                "epochs": 20,
                "batch_size": 64,
                "max_time_steps": 30,
                "experiment_name": "sa_{}_{}".format(pipeline, time.time()),
                "experiment_root_dir": None
            }
        experiment_config = experiment_boilerplate(setup)
        self.validate_experiment_dir(experiment_config)