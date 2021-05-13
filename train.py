import pytorch_lightning as pl
import torch

from data.RACEDataModule import RACEDataModule
from model.DUMAForRace import DUMAForRace


if __name__ == '__main__':
    model = DUMAForRace(
        pretrained_model='bert-large-uncased',
        learning_rate=2e-5,
        num_train_epochs=20,
        train_batch_size=16,
        train_all=True,
        use_bert_adam=True,
    )
    dm = RACEDataModule(
        model_name_or_path='bert-large-uncased',
        datasets_loader='race',
        train_batch_size=16,
        max_seq_length=128,
        num_workers=8,
        num_preprocess_processes=48,
    )
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        max_epochs=1,
        plugins='ddp_sharded',
        val_check_interval=0.2,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
