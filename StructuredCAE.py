import sys, os
sys.path.append("/Users/maxime/IndividualProject/code/Data_Assimilation/src/")

from UnstructuredCAEDA.settings.models.resNeXt import ResStack3
from UnstructuredCAEDA.settings.models.CLIC import CLIC
from UnstructuredCAEDA.train import TrainAE
from UnstructuredCAEDA.VarDA import BatchDA

from UnstructuredCAEDA.utils.expt_config import ExptConfigTest
from datetime import datetime
import argparse

TEST = True
GPU_DEVICE = 0
exp_base = ""

#global variables for DA and training:
class ExptConfig():
    EPOCHS = 5
    SMALL_DEBUG_DOM = False #For training
    calc_DA_MAE = True
    num_epochs_cv = 0
    LR = 0.0002
    print_every = 10
    test_every = 10

def main():
    print("\n-------------------------Started at {} ------------------- \n".format(datetime.now()))
    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    expdir = "experiments/structuredCAE/"
    CLIC_kwargs = {"model_name": "Tucodec", "block_type": "NeXt", "Cstd": 64, "dim": 3, "clusterInputSize": 0, "name": None}

    settings = CLIC(**CLIC_kwargs)
    settings.GPU_DEVICE = GPU_DEVICE
    settings.export_env_vars()
    print("settings data path ", settings.DATA_FP)
    
    print("------------------------- Simulation started at {} ------------------- \n\n".format(datetime.now()))
    trainer = TrainAE(settings, expdir)
    expdir = trainer.expdir #get full path
    model = trainer.train(num_epochs=expt.EPOCHS, num_workers=0, test_every=expt.test_every,
                          num_epochs_cv=expt.num_epochs_cv,
                          learning_rate = expt.LR,
                          small_debug=expt.SMALL_DEBUG_DOM)   #this will take approximately 8 hrs on a K80

    # #evaluate DA on the test set:
    results_df = BatchDA(settings, AEModel=model).run()

    print("------------------------- Ended at {} -------------------- \n".format(datetime.now()))


if __name__ == "__main__":
    main()


