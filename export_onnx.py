import argparse
import json
import sys
from pathlib import Path

import torch
import torch.onnx as onnx


PROJECT_ROOT = Path(__file__).resolve().parent
NEURAL_NETWORK_DIR = PROJECT_ROOT / "Neural_Network"

sys.path.insert(0, str(NEURAL_NETWORK_DIR))


from Neural_Network.cnn_net_prep import CNN


def load_model(path: Path) -> torch.nn.Module:
    checkpoint = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        print("Using config.json")
        state_dict = checkpoint.get("state_dict", checkpoint)

        with open(NEURAL_NETWORK_DIR / "config.json", encoding="utf-8") as file:
            config = json.load(file)

        original_cwd = Path.cwd()
        try:
            # CNN's configuration loader expects to run from Neural_Network.
            import os
            os.chdir(NEURAL_NETWORK_DIR)
            model = CNN(config["model_settings"])
        finally:
            os.chdir(original_cwd)

        model.load_state_dict(state_dict)
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    model.eval()
    return model


def main(path: str) -> None:
    model_path = Path(path).resolve()
    model = load_model(model_path)
    dummy_input = torch.randn(1, 1, 64, 100)

    onnx.export(
        model,
        (dummy_input,),
        model_path.with_suffix(".onnx"),
        opset_version=18,
        dynamo=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        default=str(PROJECT_ROOT / "models" / "model_torch_2.pt"),
    )
    args = parser.parse_args()
    main(args.model)


