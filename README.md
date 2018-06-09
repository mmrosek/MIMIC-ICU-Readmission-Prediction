# NLP for healthcare
## Predicting 30-day ICU readmissions using structured and unstructured data in MIMIC III

# Data Processing #

## Structured data
* The ETL process for structured network can be found in the *Structured* directory in *structured_etl_part1.scala* and *structured_etl_part2.py*

## Unstructured data
* All data processing scripts for unstructured data are contained in the *dataproc* directory.
* Process NOTEEVENTS to get word vectors using *data_processing_script.py*.
    * Write Discharge summaries using *get_discharge_summaries.py*
    * Build vocab from discharge summaries using *build_vocab.py*.
    * Train word embeddings on all words using *word_embeddings.py*.
    * Write trained word embeddings with our vocab using *gensim_to_embeddings* method in *extract_wvs.py*.



# Modeling #

## Structured Network
* The training of the structured network occurrs within *struc_net.py* (*Structured* directory)
* Random hyperparameter search is performed through the wrapper *py_train_struc.py*. 
* A single hyperparameter configuration can be run through *train_struc.sh*. The parameters to pass to this script can be found at the bottom of *struc_net.py*.

## Text Network
* Model can be found in the *models* directory--*models_conv_enc.py*
* Model is trained by *training_conv_encoder.py* in the *training* directory.
* Random hyperparameter search is performed through the wrapper *py_train_conv.py*.
* A single hyperparameter configuration can be run through *train_conv_enc.sh*. The parameters to pass to this script can be found at the bottom of *training_conv_encoder.py*.


## MMNet
* Model can be found in the *models* directory--*models_mmnet.py*
* Model is trained by *training_mmnet.py* in the *training* directory.
* Random hyperparameter search is performed through the wrapper *py_train_mmnet.py*.
* A single hyperparameter configuration can be run through *train_mmnet.sh*. The parameters to pass to this script can be found at the bottom of *training_mmnet.py*.

Built on https://github.com/jamesmullenbach/caml-mimic
