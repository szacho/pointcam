import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from pointcam.models import PointCAM
from pointcam.utils.builders import get_datamodule


def evaluate_linear(train_X, train_y, test_X, random_state=None):
    clf = LinearSVC(C=0.005, random_state=random_state)
    clf.fit(train_X, train_y)

    return clf.predict(test_X)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    args = parser.parse_args()

    model = PointCAM.from_exported_checkpoint(args.checkpoint_path)
    model.eval().to(DEVICE)

    datamodule = get_datamodule(model.cfg.data)
    datamodule.setup()

    scores = []
    for run_idx in range(10):
        randomize = run_idx > 0
        datamodule.flip_random(randomize)
        pl.seed_everything(42 if not randomize else run_idx)

        train_X, train_y = model.extract_features(datamodule.train_linear_dataloader())
        test_X, test_y = model.extract_features(datamodule.test_linear_dataloader())

        predictions = evaluate_linear(
            train_X, train_y, test_X, random_state=42 if randomize else None
        )

        acc = accuracy_score(test_y, predictions)
        scores.append(acc)

        print(f"Accuracy {run_idx}: {acc:.2%}")

    print(f"Mean accuracy: {np.mean(scores):.2%} ± {np.std(scores):.2%}")
