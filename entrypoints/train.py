import sys
import yaml
import torch
from transformer_playground.models import Transformer
from transformer_playground.data import OthelloDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(yaml_path: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    model = Transformer(**config["model"])
    dataset = OthelloDataset(**config["dataset"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    opt = Adam(model.parameters(), lr=config["lr"])
    device = torch.device(config["device"])
    model.to(device)
    for epoch in range(config["epochs"]):
        logger.info(f"Epoch: {epoch}")
        for idx, batch in enumerate(dataloader):
            opt.zero_grad()
            loss, _ = model.train_forward(*batch)
            logger.info(f"Batch: {idx} loss: {loss.item()}")
            loss.backward()
            opt.step()


if __name__ == "__main__":
    train(sys.argv[1])
