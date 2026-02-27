from typing import List

from dotenv import load_dotenv

from src.Transform.tranformators import apply_image_transform_noscramble
from src.model_training.Metrics import Metrics
from src.model_training.metric_functions import all_statistics
import torch

from src.utils.training_config import TestConfig, TrainingConfig
from src.utils.training_setup import load_and_prepare_test
from src.data_postprocessing.model_testing import plot_cm
from src.model_training.Metrics import Metrics
class ModelStatistics:
    def __init__(self, model_folder, model_name, model_filename, device, criterion):
        # Variables
        self.model_name = model_name
        model_filepath = f"{model_folder}/{model_filename}.pth"
        metrics_filepath = f"{model_folder}/{model_filename}.metrics"
        config_filepath = f"{model_folder}/training_config.pt"

        # initialize the training config
        self.training_config = TrainingConfig.load(config_filepath)
        # initialize the model
        model = self.training_config.model_class(**self.training_config.model_args)
        # import weights into the model
        model.load_state_dict(torch.load(model_filepath, map_location=device))

        # Load metrics
        self.metrics: Metrics = Metrics()
        self.metrics.load_metrics(metrics_filepath)

        # Calculate statistics
        stat_dir = all_statistics(model=model, dataloader=test_loader, criterion=criterion, device=device)
        self.accuracy, self.loss, self.precision, self.recall, self.f1_score, self.cm = stat_dir.values()


        # get flops and  params
        self.flops = model.get_flops()
        self.param_count = model.count_parameters()
        self.vram_params, self.vram_batch = model.get_vram_usage()


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
    def print_parameters_and_flops(self):
        col_widths = 20, 15, 15, 15
        # print header
        print(f"{'Model':<{col_widths[0]}}{'Parameters':<{col_widths[1]}}{'GFLOPs':<{col_widths[2]}}{'VRAM (GB)':<{col_widths[3]}}")
        for model_stat in self.model_stats_list:
            # convert flops to gflops with 4 decimal places
            total_gflops = model_stat.flops.total() / 1e9
            # format total_gflops to 4 decimal places
            total_gflops_formatted = f"{total_gflops:.4f}"
            vram_gb = f"{model_stat.vram_params / (1024 ** 3):.4f} + {model_stat.vram_batch / (1024 ** 3):.4f} per batch"
            print(
                f"{model_stat.model_name:<{col_widths[0]}}{model_stat.param_count:<{col_widths[1]}}{total_gflops_formatted:<{col_widths[2]}}{vram_gb:<{col_widths[3]}}")
    def plot_metrics(self, path="./output/comparison_plots/"):
        """
        Plot the training, validation metrics for all models in the comparison
        2x2 grid: training loss, validation loss, training accuracy, validation accuracy
        """
        metrics: dict[str, Metrics] = {}
        # list metrics and names
        for model_stat in self.model_stats_list:
            metrics[model_stat.model_name] = model_stat.metrics
        Metrics.compare_metrics(metrics=list(metrics.values()), names=list(metrics.keys()), path=path)





if __name__ == "__main__":
    load_dotenv()

    dcnn = ("./output/models/CustomDenseNet_Randomsplit_Dataset/20260203-001111", "CDNN","CustomDenseNet_Randomsplit_Dataset")
    scnn = ("./output/models/SimpleCNN_Randomsplit_Dataset/20260203-001401", "SCNN", "SimpleCNN_Randomsplit_Dataset")
    onelayer = ("./output/models/OneLayerNet_Randomsplit_Dataset/20260203-001342", "OneLayer", "OneLayerNet_Randomsplit_Dataset")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    test_config = TestConfig(
        batch_size=32,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset",
        dataset_size=0.15,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=32,
        transform_num_proc=8
    )
    models = [dcnn, scnn, onelayer]
    test_loader = load_and_prepare_test(test_config)
    model_stats_list = []
    for model_filepath, model_name, model_filename in models:
        model_stats = ModelStatistics(
            model_folder=model_filepath,
            model_name=model_name,
            model_filename=model_filename,
            device=device,
            criterion=criterion
        )
        model_stats_list.append(model_stats)
    mc = ModelComparison(model_stats_list)
    mc.print_table()
    path = "./output/comparison_plots/comparison_metrics.png"
    mc.plot_metrics(path=path)
    mc.print_parameters_and_flops()