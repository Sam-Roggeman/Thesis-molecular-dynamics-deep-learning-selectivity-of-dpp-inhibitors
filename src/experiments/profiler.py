from torch.profiler import profile, ProfilerActivity
import torch

from src.Models.SimpleCNN import SimpleCNN
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.data_loading import HFDataloader
from src.utils.training_config import TrainingConfig 
import os


def profile_model(model, dataloader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_iter = iter(dataloader)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
        for _ in range(10):  # Profile a few steps
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                batch = next(train_iter)
            model.train()
            inputs, labels = batch["data"], batch["labels"]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()

    # Analyze the results
    print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))

    # Export for detailed analysis
    prof.export_chrome_trace("trace.json")  
        
if __name__ == "__main__":
    # load dotenv variables for HuggingFace API token
    from dotenv import load_dotenv
    load_dotenv()
    
    config = TrainingConfig(
        model_class=SimpleCNN,
        model_args={
            "input_size": 168,
            "dropout_rate": 0.2
        },
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        training_transform=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        batch_size=128,
    )
    # Initialize model
    model = config.model_class(**config.model_args)
    dataloader = HFDataloader.initialize_dataloaders(config)["train"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Get a batch of data
    prof = profile_model(model, dataloader, criterion, optimizer)
    
    # Analyze the results
    print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))

    # Export trace to a file for detailed analysis in Chrome Trace Viewer
    model_name = config.model_class.__name__ 
    # ensure the output directory exists
    os.makedirs("/project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings/output/profiling", exist_ok=True)
    filepath = f"/project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings/output/profiling/{model_name}_trace.json"
    prof.export_chrome_trace(filepath)
    print(f"Trace exported to {filepath}")
    