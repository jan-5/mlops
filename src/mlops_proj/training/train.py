import wandb
import lightning as L
import os
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.mlops_proj.data.glue_data_module import GLUEDataModule
from src.mlops_proj.models.glue_transformer import GLUETransformer


def train(args):
    
    epochs = 3
    seed = 42
    accelerator = "auto"
    devices = 1
    project_name = os.environ.get("WANDB_PROJECT", "mlops_proj")
    
    run_name = (
        f"lr{args.learning_rate:.0e}_"
        f"ls-{args.lr_schedule_type[:3]}_"
        f"ws{args.warmup_steps}_"
        f"wd{args.weight_decay}_"
        f"tb{args.train_batch_size}_"
        f"eb{args.eval_batch_size}_"
        f"gc{args.gradient_clip_val}"
    )

    wandb.login()
    
    wandb_dir = os.path.join(os.getcwd(), "experiments", "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    
    logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=wandb_dir,
        resume="never"
    )

    L.seed_everything(seed)

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        lr_schedule_type=args.lr_schedule_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_clip_val=args.gradient_clip_val,
    )

    logger.experiment.config.update({
        "model_name": model.hparams.model_name_or_path,
        "task": dm.task_name,
        "learning_rate": model.hparams.learning_rate,
        "lr_schedule_type": model.hparams.lr_schedule_type,
        "warmup_steps": model.hparams.warmup_steps,
        "weight_decay": model.hparams.weight_decay,
        "train_batch_size": model.hparams.train_batch_size,
        "eval_batch_size": model.hparams.eval_batch_size,
        "gradient_clip_val": model.hparams.gradient_clip_val,
        "max_seq_length": dm.max_seq_length,
        "epochs": epochs,
        "seed": seed,
    })
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ckpt_best_f1 = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{run_name}-best-f1-{{epoch:02d}}-{{f1:.4f}}",
        monitor="f1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    ckpt_best_acc = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{run_name}-best-acc-{{epoch:02d}}-{{accuracy:.4f}}",
        monitor="accuracy",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    ckpt_best_loss = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{run_name}-best-loss-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    ckpt_last = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{run_name}-last",
        save_last=True,
        auto_insert_metric_name=False,
    )
 
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        gradient_clip_val=model.hparams.gradient_clip_val,
        callbacks=[ckpt_best_f1, ckpt_best_acc, ckpt_best_loss, ckpt_last],
    )
    trainer.fit(model, datamodule=dm)

    wandb.finish()