# driver
import logging, torch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CNN_Training")


if __name__ == "__main__":
    logger.info("Starting the test application")
    if torch.cat is torch.cat:
        logger.info("torch.cat is the same as torch.cat")
    func = lambda tensor: tensor.view(tensor.shape[0], -1)
    x = torch.tensor([[[[1, 2], [3, 4]]]])
    y = torch.tensor([[5,6]])
    print(func(x))
    print(torch.cat([func(x), y], dim=1))

