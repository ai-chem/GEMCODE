## Files

- The `GAN` folder contains the folders 
    - `CCDC_fine-tuning`: dataset of coformers for pre-training the generative model, GAN checkpoint and notebook to start the training
    - `ChEMBL_training`: main dataset with molecules for basic training and notebook to start the training
    - `scripts`: contains files for initialisation of the GAN model

- The `TVAE` folder contains the folders:
    - `generate`: necessary scripts to generate molecules by an already trained model based on the Transformer architecture
    - `train_vae`, `train_cvae`: necessary scripts for training TVAE and TCVAE models
    - Also contains other necessary files for training a generative model based on Transformer architecture.

- The `language_models` folder contains the folders:
    - `gpt2`: code for generating coformers using GPT2
    - `llama3-8`: code for generating coformers using Llama-3-8B