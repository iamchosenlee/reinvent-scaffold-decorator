Implementation of the SMILES-based scaffold decorator used in "SMILES-based deep generative scaffold decorator for de-novo drug design"
=======================================================================================================================================

This repository holds all the code used to create, train and sample a SMILES-based scaffold decorator described in [SMILES-based deep generative scaffold decorator for de-novo drug design](https://chemrxiv.org/articles/SMILES-Based_Deep_Generative_Scaffold_Decorator_for_De-Novo_Drug_Design/11638383). Additionally, it contains the code for pre-processing the training set, as explained in the manuscript. 

The scripts and folders are the following:

1) Python files in the main folder are all scripts. Run them with `-h` for usage information.
2) `./training_sets` folder: The two molecular sets used in the manuscript, separated between training and validations sets.
3) `./trained_models` folder: One already trained model for both the DRD2 and ChEMBL models.

### Added Features and Differences
* Add `reinforcement.py` for molecular property optimization with REINFORCE algorithm. Run `python reinforcement.py` for optimization with default setting(Optimize the binding affinity with JAK2).
* Add `reinforce.ipynb` as a jupyter notebook version of `reinforcement.py`.
* Add `join.py` for quick sampling and joining scaffold-decorator into a molecule without using Spark. See function `fast_join(input_scaffolds, sample_model)`.

### Optimization Results
The log of one iteration will look like this.
~~~~
$> python reinforcement.py
Loading pretrained parameter "encoder.encoder.0.cached_zero_vector".
Loading pretrained parameter "encoder.encoder.0.W_i.weight".
Loading pretrained parameter "encoder.encoder.0.W_h.weight".
Loading pretrained parameter "encoder.encoder.0.W_o.weight".
Loading pretrained parameter "encoder.encoder.0.W_o.bias".
Loading pretrained parameter "ffn.1.weight".
Loading pretrained parameter "ffn.1.bias".
Moving model to cuda
Generating molecules...: : 1195it [01:38, 12.17it/s]                                                                                                         
Mean value of predictions: 7.756021530084555                                                                                                                 
reinforcement.py:209: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
['C#CCCN1CCCC(Nc2nc(C)ccc2-c2cnc(C(C)O)c(-c3ccccc3)n2)C1', 'C#CCCOC(=O)N1CCC(n2cc(C(N)=O)c(Nc3ccc(F)cc3)n2)C(C#N)C1', 'C=C(C#N)C(=O)Nc1ccc(-c2ncnc3c2cc(C(=O)OCCN2CCOCC2)n3C)cc1', 'C=C(C)C(=O)Nc1cc(-n2c(=O)ccc3cnc4ccc(-c5ccncc5)cc4c32)ccc1C', 'C=C(C)C(=O)Nc1ccc(Cl)c(-c2ncnc3c2cc(C(=O)NCCN2CCCC2)n3C)c1', 'C=C(C)C(=O)Nc1cccc2c(-c3nc(NC(=O)C4CC4)ncc3C)c[nH]c12', 'C=C(C)C(=O)Nc1cccc2c(-c3nc(NC(C)CN4CCN(C)CC4)ncc3C)c[nH]c12', 'C=C(C)C(=O)Nc1ccccc1-c1ncnc2[nH]cc(CCOC)c12', 'C=C(C)CC(=O)N1CCN(c2cccc(C3=C(c4c[nH]c5ccccc45)C(=O)NC3=O)c2)CC1', 'C=C(C)NC(=O)Nc1cc(-n2c(=O)ccc3cnc4ccc(-c5cccnc5)cc4c32)ccc1C'] [7.77277353 7.17321644 6.73085678 6.19672391 5.99296015 6.74218652
 6.21127877 7.59518544 6.68516158 6.30845166]
#################################################################################################
Policy gradient...:   0%|                                                                                                             | 0/15 [00:00<?, ?it/s0th cur_reward, cur_loss:  60.088469888818814 867.7528686523438                                                                                               
Policy gradient...:   7%|██████▋                                                                                              | 1/15 [00:01<00:27,  1.99s/it1th cur_reward, cur_loss:  42.73736631510332 218.4180908203125                                                                                                
Policy gradient...:  13%|█████████████▍                                                                                       | 2/15 [00:04<00:27,  2.13s/it2th cur_reward, cur_loss:  48.55983039492834 691.9469604492188                                                                                                
Policy gradient...:  20%|████████████████████▏                                                                                | 3/15 [00:06<00:24,  2.05s/it3th cur_reward, cur_loss:  44.54524701185346 305.70098876953125                                                                                               
Policy gradient...:  27%|██████████████████████████▉                                                                          | 4/15 [00:08<00:22,  2.09s/it4th cur_reward, cur_loss:  68.98006616479421 697.1046142578125                                                                                                
Policy gradient...:  33%|█████████████████████████████████▋                                                                   | 5/15 [00:12<00:26,  2.66s/it5th cur_reward, cur_loss:  45.236622984034625 1890.140625                                                                                                     
Policy gradient...:  40%|████████████████████████████████████████▍                                                            | 6/15 [00:13<00:21,  2.40s/it6th cur_reward, cur_loss:  53.712883575689006 364.78167724609375                                                                                              
Policy gradient...:  47%|███████████████████████████████████████████████▏                                                     | 7/15 [00:15<00:17,  2.17s/it7th cur_reward, cur_loss:  57.72964258873409 1429.58984375                                                                                                    
Policy gradient...:  53%|█████████████████████████████████████████████████████▊                                               | 8/15 [00:18<00:15,  2.29s/it8th cur_reward, cur_loss:  49.37922326367231 1632.970947265625                                                                                                
Policy gradient...:  60%|████████████████████████████████████████████████████████████▌                                        | 9/15 [00:20<00:13,  2.19s/it9th cur_reward, cur_loss:  51.87714441513715 434.890380859375                                                                                                 
Policy gradient...:  67%|██████████████████████████████████████████████████████████████████▋                                 | 10/15 [00:22<00:10,  2.13s/it10th cur_reward, cur_loss:  38.78872680780572 521.468017578125                                                                                                
Policy gradient...:  73%|█████████████████████████████████████████████████████████████████████████▎                          | 11/15 [00:24<00:08,  2.18s/it11th cur_reward, cur_loss:  46.247302487908584 1818.3994140625                                                                                                
Policy gradient...:  80%|████████████████████████████████████████████████████████████████████████████████                    | 12/15 [00:27<00:07,  2.51s/it12th cur_reward, cur_loss:  49.264396263668786 535.4310302734375                                                                                              
Policy gradient...:  87%|██████████████████████████████████████████████████████████████████████████████████████▋             | 13/15 [00:30<00:05,  2.52s/it13th cur_reward, cur_loss:  54.47546311090269 1544.2960205078125                                                                                              
Policy gradient...:  93%|█████████████████████████████████████████████████████████████████████████████████████████████▎      | 14/15 [00:32<00:02,  2.53s/it14th cur_reward, cur_loss:  58.170570312714574 479.22235107421875                                                                                             
Policy gradient...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:34<00:00,  2.32s/it]
reinforcement.py:338: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
Generating molecules...: : 11it [00:02,  5.21it/s]                                                                                                           
Mean value of predictions: 8.070050488896452                                                                                                                 
reinforcement.py:209: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
0th Sample trajectories:
C=CC(=O)N1CCCC(n2cc(-c3nn(CC(=O)C(=O)N4CCOC4)c4ncnc(N)c34)cn2)C1=O
C=CS(=O)(=O)N1CC(c2nn(Cc3cc(C#N)cnc3N(C)C)c3ncnc(N)c23)C1
CC(C)(C)c1ccc(F)c(-c2nn(C3CCN(C(=O)CC#N)CC3)c3ncnc(N)c23)n1
COCCn1cc(-c2nn(C3CCN(C(=O)C(F)(F)F)CC3)c3ncnc(N)c23)cn1
N#CC1CC(N2CC2(O)F)CCCC1n1cc(C(N)=O)c(Nc2ccnc(F)c2)n1
#################################################################################################

~~~~
<br/>
<br/>
This is an example of generated molecules with optimization. <br/>
<p align="center">
  <img src="https://user-images.githubusercontent.com/29084981/128223568-4f92a595-2ad7-45b0-903c-4ae0834f4e47.png">
</p>


Requirements
------------
The repository includes a Conda `environment.yml` file with the required libraries to run all the scripts. In some scripts Spark 2.4 is required (and thus Java 8) and by default should run in local mode without any issues. For more complex set-ups, please refer to the [Spark documentation](http://spark.apache.org/docs/2.4.3/). All models were tested on Linux with both a Tesla V-100 and a Geforce 2070. It should work just fine with other Linux setups and a mid-high range GPU.

Install
-------
A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.

~~~~
$> git clone <repo url>
$> cd <repo folder>
$> conda env create -f environment.yml
$> conda activate reinvent-scaffold-decorator
(reinvent-scaffold-decorator) $> ...
~~~~

From here the general usage applies.

General Usage
-------------
Several tools are supplied and are available as scripts in the main folder. Further information about the tool's arguments, please run it with `-h`. All output files are in tsv format (the separator is \t).

### Preprocessing the training sets

Any arbitrary molecular set has to be pre-processed before being used as a training set for a decorator model. This process is done in two steps:

1) Slice (`slice_db.py`): This script accepts as input a SMILES file and it exhaustively slices given a set of slicing rules (Hussain-Rea, RECAP), a maximum number of attachment points and a set of conditions (see `conditions.json.example` for more information). Rules can be easily extended and new sets of conditions added. This script can output a SMILES file with (scaffold, dec1;dec2;...) which can be used in the next step and/or a parquet file with additional information.
2) Create randomized SMILES (`create_randomized_smiles.py`): From the SMILES output of the first file, several randomized SMILES representations of the training set must be generated. Depending whether a single-step or a multi-step decorator model is wanted, different files are generated.

Notice that both scripts use Spark.

### Creating, training and sampling decorator models

This code enables training of both single-step and multi-step decorator models. The training process is exactly the same, only changing the training set pre-processing.

1) Create Model (`create_model.py`): Creates a blank model file.
2) Train Model (`train_model.py`): Trains the model with the specified parameters. Tensorboard data may be generated.
3) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of decorations given a list of scaffolds. It can also retrieve the log-likelihood in the process. Notice that this is not the preferred way of sampling the model (see below).
4) Calculate NLL (`calculate_nlls.py`): Requires as input a TSV file with scaffolds and decorations and outputs the same list with the NLL calculated for each one.

### Exhaustively decorating scaffolds

A special script (`sample_scaffolds.py`) to exhaustively generate a large number of decorations is supplied. This scripts can be used with both single-step and multi-step models. Notice that this script requires **both** a GPU and Spark. It works the following way:

0) A SMILES file with the scaffolds with the attachment points numbered (`[*]` -> `[*:N]`, where N is a number from 0 to the number of attachment points.
1) Generates at most `r` randomized SMILES of each scaffold (`-r` to change how many SMILES are generated at each round).
2) Samples `n` times each randomized SMILES generated in the previous step (`-n`to change the value).
3) Joins the scaffolds with the generated decorations and removes duplicates/invalids.
4) In the case of the single-step, nothing more is necessary, but in the multi-step model, a loop starting at step 1 is repeated until everything is fully decorated.
5) Everything, including the half-decorated molecules, is written down in a parquet file (or a CSV file, if the option `--output-format=csv` is used instead) for further analysis. The results have to be then extracted from the parquet/CSV file (i.e. by extracting SMILES that have the * token, for instance).

**CAUTION:** Large `n` and `r`parameters should be used for the single-step decorator model (for instance `r=2048` and `n=4096`). In the case of the multi-step model, very low values should be used instead (e.g. `r=16` and `n=32`).
**NOTICE:** A new option was added to allow using repeated randomized SMILES (`--repeated-randomized-smiles`). It is disabled by default.

Usage examples
--------------

Create the DRD2 dataset as described in the manuscript.
~~~~
(reinvent-scaffold-decorator) $> mkdir -p drd2_decorator/models
(reinvent-scaffold-decorator) $> ./slice_db.py -i training_sets/drd2.excapedb.smi.gz -u drd2_decorator/excape.drd2.hr.smi -s hr -f conditions.json.example
(reinvent-scaffold-decorator) $> ./create_randomized_smiles.py -i drd2_decorator/excape.drd2.hr.smi -o drd2_decorator/training -n 50 -d multi
~~~~
Train the DRD2 model using the training set created before.
~~~~
(reinvent-scaffold-decorator) $> ./create_model.py -i drd2_decorator/training/001.smi -o drd2_decorator/models/model.empty -d 0.2
(reinvent-scaffold-decorator) $> ./train_model.py -i drd2_decorator/models/model.empty -o drd2_decorator/models/model.trained -s drd2_decorator/training -e 50 -b 64 
~~~~
Sample one scaffold exhaustively.
~~~~
(reinvent-scaffold-decorator) $> echo "[*:0]CC=CCN1CCN(c2cccc(Cl)c2[*:1])CC1" > scaffold.smi
(reinvent-scaffold-decorator) $> spark-submit --driver-memory=8g sample_scaffolds.py -m drd2_decorator/models/model.trained.50 -i scaffold.smi -o generated_molecules.parquet -r 16 -n 16 -d multi
~~~~

**Notice**: To change it to a single-step model, the `-d single` option must be used in all cases where `-d multi` appears.
**Caution**: Spark run in local mode generally has a default of 1g of memory. This can be insufficient in some cases. That is why we use `spark-submit` to run the last script. Please change the --driver-memory=XXg to a suitable value. If you get out of memoy errors in any other script, also use the spark-submit trick.


