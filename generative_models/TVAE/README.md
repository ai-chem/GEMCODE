## Instructions for running generation.

In order to set up a model for generation, you must have the weights of the trained model and use the
```AAAI_code/TVAE/generate.py``` function.

```AAAI_code/TVAE/TVAE/generate/example.py``` This is sample code to run molecule generation. 

The generator function generates the specified number of molecules.

In the folder ```AAAI_code/TVAE/TVAE/generate/for_generation``` copy the model weights you want to use 
for generation

```-n_samples``` - number of molecules to be generated.

```-path_to_save``` - suffix to the file path to save the molecules. It is necessary to give a name to the file for generation.

```-save ```- whether to save the generated molecules? True/False

```-spec_conds``` - None for random assignment of physical properties/
list for specifying molecules of interest. Example: [1,1,0].


## Instructions for running training.
For training you need to run the file train_phys.py, this file lies in the dirrectories: 
```
AAAI_code/TVAE/train_cVAE,
AAAI_code/TVAE/train_VAE,
AAAI_code/TVAE/train_cVAE_sgdr.
```
depending on the model configuration. (train_cVAE) is most optimal for the coformer task.

Then, for fine-tune the model, there is a folder in the directories mentioned above 
```
.../fine_tune....
```
This folder may have different names, as the title reflected information about the core learning, and usually,
the number of the epoch of the main training in which the most optimal training results were obtained.
The content of the fine_tune folder is duplicated with the main folder, with one difference that the ``old_weights`` folder must contain weights
of the main model and files ``SRC.pkl`` and ``TRG.pkl``, in which the dictionary for creating embbedding of molecules is written. These files need not be transferred from the main training, since the dictionary for the large molecule dataset will be larger, and the weights for the model are customised to them. (It does not matter that the dictionary in the small dataset is smaller - any new molecules in the small dataset will be embbedding using the existing dictionary).
It is also worth noting that the ```train_phys.py``` file for fine-tune requires additional settings, such as, 
```-load_weights``` argument specifying the path where the weights for initialising the model are located 
(by default this is the ```old_weights``` folder).

## Available parameters of the train_phys.py file.
This file describes the training pipeline and the parsing of arguments for training from the console using the
using ```argparse.ArgumentParser()```. This is useful when running training from the console on the server to specify the 
flags.
However, this whole design is very cumbersome, and I plan to translate this miracle into some json or other 
config option. Otherwise, it should work.

The main arguments to pay attention to when configuring the training run are:

```-lr_scheduler``` - select the type of LR scheduler. sgdr or warmup. However, warmup is hard to configure, if there is a need to
to form your own scheduler with warmup, you should look for another solution.

```-lr``` is the starting learning rate for.

```-epochs``` - How many epochs of learning to teach the model.

```-load_weights``` - Path to the folder with weights for initialisation. And whether to load them at all.

```-checkpoint ```- Specifies the interval in which the weights of the model are saved during training. 
And if None, it is not saved at all.

Model structure settings:
```-latent_dim``` - size of latent space, between encoder and decoder.

```-cond_dim``` - size of the property vector. We have used 3, but if necessary they can be 
expanded if necessary.

```-d_model``` - the dimensionality within the model.

```-n_layers``` - the number of encoders and decoders going after each other.

```-heads``` - number of heads in the model.

```-batchsize``` - The size of the patch. However, it doesn't quite work, as this case uses the 
''effective batch'' size, in which a batch with sequences of the same length is fed to the input of the model.
If -batchsize is less than the length of the "efficient batch", then the batch will be -batchsize. 
I don't recommend putting -batchsize small.

```-src_data,-src_data_te,-trg_data,-trg_data_te``` - Paths to the dataset of molecules to the encoder input,
to the input to the encoder for the test, to the input to the decoder, to the input to the decoder for the test respectively.
Despite the fact that src and trg are the same objects, but in this case src with a token is fed to the encoder input
src with tokeniser without <sos> and <eos> tokens. The data format is .txt


```-save_folder_name```- the path to the folder where to save the model weights for each epoch during training.

```-train_data_csv``` - path to the folder with training data in .csv format to check for the
novelty of the generated molecules.

```-cond_test_path,-cond_train_path``` - paths to the list of physical properties for test and training, respectively.
.csv file format


## Where to get scales for CVAE/VAE models and which ones to use?
To create a model, you need ``model_weights`` scales, which can be downloaded from repo using the link:

```https://filetransfer.io/data-package/MstyseWg#link```

The following folders are available in this mode:


```TVAE``` - this folder contains ```VAE``` weight models.

```TCVAE``` - this folder contains the ```CVAE`` weight models.

```GAN``` - Folder with weights of the initial generative LSTM-based GAN model.
