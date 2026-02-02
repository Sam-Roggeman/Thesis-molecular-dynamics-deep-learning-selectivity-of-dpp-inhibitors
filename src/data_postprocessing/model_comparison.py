from typing import List

from numpy.random.tests.test_randomstate import TestThread
from torch.utils.data import dataloader

from src.Transform.tranformators import apply_image_transform_noscramble
from src.model_training.Metrics import Metrics
from src.model_training.metric_functions import all_statistics
import torch

from src.utils.training_config import TestConfig, TrainingConfig
from src.utils.training_setup import load_and_prepare_test
import matplotlib.pyplot as plt


class ModelStatistics:
    def __init__(self, model_folder, model_name, model_filename, device, criterion):
        # Variables
        self.model_name = model_name
        model_filepath = f"{model_folder}/{model_filename}.pth"
        metrics_filepath = f"{model_folder}/{model_filename}.metrics"
        config_filepath = f"{model_folder}/training_config.json"

        # Load model, metrics, training_config
        model = torch.load(config_filepath, map_location=device)
        self.training_config = TrainingConfig(model_folder)
        self.metrics: Metrics = torch.load(metrics_filepath, map_location=device)

        # Calculate statistics
        stat_dir = all_statistics(model, device, criterion, device)
        self.accuracy, self.loss, self.precision, self.recall, self.f1_score = stat_dir.values()




class ModelComparison:
    def __init__(self, modelstatistics: List[ModelStatistics]):
        self.model_stats_list = modelstatistics

    def print_table(self):
        col_widths = 20, 15, 10, 10, 10, 10
        # print header
        print(
            f"{'Model':<{col_widths[0]}}{'Accuracy':<{col_widths[1]}}{'Loss':<{col_widths[2]}}{'Precision':<{col_widths[3]}}{'Recall':<{col_widths[4]}}{'F1-Score':<{col_widths[5]}}")
        for model_stat in self.model_stats_list:
            print(
                f"{model_stat.model_name:<{col_widths[0]}}{model_stat.accuracy:<{col_widths[1]}.4f}{model_stat.loss:<{col_widths[2]}.4f}{model_stat.precision:<{col_widths[3]}.4f}{model_stat.recall:<{col_widths[4]}.4f}{model_stat.f1_score:<{col_widths[5]}.4f}")





if __name__ == "__main__":
    model1 = ("../models/OneLayerNet/OneLayerNet_Randomsplit_Dataset/20251212-200554/",
              "OneLayer")
    model2 = ("../models/CustomDenseNet/CustomDenseNet_Randomsplit_Dataset/20251209-125815/",
              "CDNN")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    test_config = TestConfig(
        batch_size=32,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset",
        hf_token=huggingface_token,
        dataset_size=0.15,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=32,
        transform_num_proc=8,
        cache_folder="/project_scratch/dataset_cache/"
    )
    test_loader = load_and_prepare_test(test_config)
    model_stats_list = []
    for model_filepath, model_name in [model1, model2]:
        model_stats = ModelStatistics(
            model_folder=model_filepath,
            model_name=model_name,
            model_filename="CustomDenseNet_Randomsplit_Dataset",
            device=device,
            criterion=criterion
        )
        model_stats_list.append(model_stats)
