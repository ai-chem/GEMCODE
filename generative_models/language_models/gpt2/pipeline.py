import os

import smiles_gpt.smiles_gpt as gpt
import torch
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import GPT2Config, GPT2LMHeadModel

pret_dataset = "data\\database_ChEMBL.txt"
tune_dataset = "data\\cocrys.txt"
checkpoints_dir = "smiles_gpt\\checkpoints"
tokenizer_filename = f"{checkpoints_dir}\\benchmark-10m\\tokenizer.json"
pret_ckpt_name = "pret_chembl"
tune_ckpt_name = "tune_chembl"

# Tokenizer, model, optimizer, scheduler, and trainer hyperparameters.
hyperparams = {
    "batch_size": 128,
    "max_epochs": 15,
    "min_epochs": 5,
    "max_length": 512,
    "learning_rate": 1e-4,
    "weight_decay": 0.001,
    "adam_eps": 1e-8,
    "adam_betas": (0.9, 0.999),
    "scheduler_T_max": 150_000,
    "final_learning_rate": 5e-8,
    "vocab_size": 1_000,
    "min_frequency": 2,
    "top_p": 0.96,
    "n_layer": 6,
    "n_head": 12,
    "n_embd": 12 * 48,
}

gpus = 1
num_workers = 6
is_tokenizer_pretrained = True


tokenizer = gpt.SMILESBPETokenizer(dropout=None)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = tokenizer.from_file(
    f"{checkpoints_dir}\\benchmark-10m\\vocab.json",
    f"{checkpoints_dir}\\benchmark-10m\\merges.txt",
)

tokenizer


tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
    tokenizer_filename,
    model_max_length=hyperparams["max_length"],
)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_layer=hyperparams["n_layer"],
    n_head=hyperparams["n_head"],
    n_embd=hyperparams["n_embd"],
    n_positions=hyperparams["max_length"],
    n_ctx=hyperparams["max_length"],
)

model = GPT2LMHeadModel(config)

torch.set_float32_matmul_precision("medium")

hyperparams = {
    "batch_size": 128,
    "max_epochs": 15,
    "min_epochs": 5,
    "max_length": 512,
    "learning_rate": 1e-4,
    "weight_decay": 0.001,
    "adam_eps": 1e-8,
    "adam_betas": (0.9, 0.999),
    "scheduler_T_max": 150_000,
    "final_learning_rate": 5e-8,
    "vocab_size": 1_000,
    "min_frequency": 2,
    "top_p": 0.96,
    "n_layer": 6,
    "n_head": 12,
    "n_embd": 12 * 48,
}

if __name__ == "__main__":

    # pretrain
    datamodule = gpt.LMDataModule(
        pret_dataset,
        tokenizer,
        batch_size=hyperparams["batch_size"],
        num_workers=num_workers,
    )

    checkpoint_cb = ModelCheckpoint(f"{checkpoints_dir}/{pret_ckpt_name}/")
    early_stopping_ppl = EarlyStopping(
        monitor="ppl",
        patience=3,
        min_delta=5e-3,
        check_finite=True,
        stopping_threshold=1.1,
        divergence_threshold=hyperparams["vocab_size"] / 10,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=True,
    )

    trainer = Trainer(
        callbacks=[checkpoint_cb, early_stopping_ppl],
        max_epochs=hyperparams["max_epochs"],
        min_epochs=hyperparams["min_epochs"],
        val_check_interval=0.4,
        limit_train_batches=0.2,
        log_every_n_steps=20,
    )

    lit_model = gpt.GPT2LitModel(
        model,
        batch_size=hyperparams["batch_size"],
        learning_rate=hyperparams["learning_rate"],
        final_learning_rate=hyperparams["final_learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        adam_eps=hyperparams["adam_eps"],
        adam_betas=hyperparams["adam_betas"],
        scheduler_T_max=hyperparams["scheduler_T_max"],
    )

    trainer.fit(lit_model, datamodule)
    lit_model.transformer.save_pretrained(f"{checkpoints_dir}/{pret_ckpt_name}/")

    checkpoint_cb = ModelCheckpoint(f"{checkpoints_dir}/{tune_ckpt_name}/")
    datamodule = gpt.LMDataModule(
        tune_dataset,
        tokenizer,
        batch_size=hyperparams["batch_size"],
        num_workers=num_workers,
    )

    trainer = Trainer(
        callbacks=[checkpoint_cb, early_stopping_ppl],
        max_epochs=hyperparams["max_epochs"],
        min_epochs=hyperparams["min_epochs"],
        val_check_interval=0.4,
        limit_train_batches=0.2,
        log_every_n_steps=20,
    )

    trainer.fit(lit_model, datamodule)
    lit_model.transformer.save_pretrained(f"{checkpoints_dir}/{tune_ckpt_name}/")

    model = GPT2LMHeadModel.from_pretrained(
        f"{checkpoints_dir}\\{tune_ckpt_name}", output_attentions=True
    )

    model.eval()

    n_generated = 10_000
    generated_smiles_list = []

    for _ in tqdm.tqdm(range(n_generated)):
        # Generate from "<s>" so that the next token is arbitrary.
        smiles_start = torch.LongTensor([[tokenizer.bos_token_id]])
        # Get generated token IDs.
        generated_ids = model.generate(
            smiles_start,
            # generation_config=cfg,
            max_length=hyperparams["max_length"],
            do_sample=True,  # gd
            top_p=hyperparams["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode the IDs into tokens and remove "<s>" and "</s>".
        generated_smiles = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_smiles_list.append(generated_smiles)

    with open(f"out.txt", "w") as f:
        for s in generated_smiles_list:
            f.write(s + "\n")
