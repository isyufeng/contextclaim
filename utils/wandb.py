import wandb
import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path
import time


class WandbLogger:
    def __init__(
            self,
            project: str,
            config: Dict[str, Any],
            name: str = None,
            tags: list = None,
            notes: str = None,
            group: str = None,
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            config: Configuration dictionary
            name: Run name
            tags: Experiment tags
            notes: Experiment notes
            group: Experiment group name
        """
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
        )

        self.checkpoint_dir = Path("checkpoints") / self.run.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(
            self,
            metrics: Dict[str, float],
            step: int = None,
            commit: bool = True
    ):
        """Log training metrics."""
        self.run.log(metrics, step=step, commit=commit)

    def log_model(
            self,
            model: nn.Module,
            epoch: int,
            optimizer: torch.optim.Optimizer = None,
            metric: float = None,
    ):
        """Save model checkpoint and upload to WandB."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        metric_str = f"_{metric:.4f}" if metric is not None else ""
        checkpoint_name = f"epoch_{epoch}{metric_str}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        torch.save(checkpoint, checkpoint_path)

        artifact = wandb.Artifact(
            name=f"model-checkpoint-{self.run.id}",
            type="model",
            description=f"Model checkpoint for epoch {epoch}",
        )
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact)

    def log_batch_metrics(
            self,
            metrics: Dict[str, float],
            step: int,
            commit: bool = False
    ):
        """Log per-batch metrics."""
        batch_metrics = {f"batch/{k}": v for k, v in metrics.items()}
        self.log_metrics(batch_metrics, step, commit)

    def log_images(
            self,
            images_dict: Dict[str, torch.Tensor],
            step: int = None,
            commit: bool = True
    ):
        """Log image data."""
        wandb_images = {
            k: wandb.Image(v.cpu().numpy())
            for k, v in images_dict.items()
        }
        self.run.log(wandb_images, step=step, commit=commit)

    def log_histogram(
            self,
            name: str,
            values: torch.Tensor,
            step: int = None,
            commit: bool = True
    ):
        """Log histogram data."""
        self.run.log(
            {name: wandb.Histogram(values.cpu().numpy())},
            step=step,
            commit=commit
        )

    def finish(self):
        """Finish the experiment run."""
        self.run.finish()


def main():
    config = {
        "model": {
            "name": "resnet50",
            "pretrained": True,
            "num_classes": 10
        },
        "training": {
            "epochs": 10,
            "iterations": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam"
        }
    }

    logger = WandbLogger(
        project="my-project",
        name="experiment-001",
        config=config,
        tags=["baseline", "resnet50"],
        notes="Test ResNet50 baseline model",
        group="resnet-experiments"
    )

    global_step = 0

    try:
        for iteration in range(config["training"]["iterations"]):
            logger.log_metrics({
                "iteration": iteration,
                "iteration_start_time": time.time()
            }, step=global_step, commit=True)

            for epoch in range(config["training"]["epochs"]):
                epoch_metrics = {
                    "train/loss": 0.5 - (iteration * 0.1 + epoch * 0.05),
                    "train/accuracy": 0.8 + (iteration * 0.05 + epoch * 0.02),
                    "train/learning_rate": config["training"]["learning_rate"],
                    "current_epoch": epoch,
                    "current_iteration": iteration
                }

                logger.log_metrics(epoch_metrics, step=global_step, commit=True)

                model = torch.nn.Linear(10, 10)
                logger.log_histogram(
                    f"weights/iteration_{iteration}/layer1",
                    model.weight.data,
                    step=global_step,
                    commit=False
                )

                if epoch % 5 == 0 or epoch == config["training"]["epochs"] - 1:
                    logger.log_model(
                        model=model,
                        epoch=epoch,
                        metric=epoch_metrics["train/accuracy"]
                    )

                global_step += 1

            iteration_summary = {
                "iteration_summary/avg_loss": epoch_metrics["train/loss"],
                "iteration_summary/final_accuracy": epoch_metrics["train/accuracy"],
                "iteration_summary/duration": time.time() - logger.run.started_at
            }
            logger.log_metrics(iteration_summary, step=global_step, commit=True)

    except Exception as e:
        logger.log_metrics({"error": str(e)}, commit=True)
        raise e
    finally:
        logger.finish()


if __name__ == "__main__":
    main()