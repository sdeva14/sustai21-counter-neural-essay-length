# Sustai21: Countering the Influence of Essay Length in Neural Essay Scoring
This project contains a python implementation for the Sustai21 Workshop paper, whose title is "Countering the Influence of Essay Length in Neural Essay Scoring".

## Requirements

#### Conda environment
We recommend using Conda environment for setup. It is easy to build an environment by the provided environment file. It is also possible to setup manually by considering the information in "spec-file.txt". 

Our environment file is built based on CUDA9 driver and corresponding libraries, thus PyTorch and other libaries should be managed by the target GPU environment. Otherwise, GPU flag should be disabled as a library. For the variation of XLNet, we use Transformers library implemented by Huggingface (Wolf et al, 2019).

    conda create --name py3_torch_cuda9 --file spec-file.txt
    source activate py3_torch_cuda9
    pip install pytorch-transformers==2.3.0

#### Dataset and materials
Dataset and pretrained embedding cannot be attached in the submission due to large size, thus it should be downloaded from the Github of previous work.

- Dataset: The ASAP dataset is available at Kaggle official page in public. Recent neural essay scoring systems discussed in our paper use the 5-fold partition proposed in (Taghipour and Ng, 2016). It should be downloaded from the Github, and the location should be configured in "build_config.py" with "--data_dir" option.
The TOEFL dataset is available according to the link in the original paper (Blanchard et al. 2013) with LDC license. 

ASAP CV partition link: https://github.com/nusnlp/nea

TOEFL dataset link: https://catalog.ldc.upenn.edu/LDC2014T06

TOEFL dataset split: https://github.com/sdeva14/emnlp20-centering-neural-hds

- Pretrained embedding: Previous neural essay scoring systems are based on the pretrained embedding proposed in (Taghipour and Ng, 2016), it is available at their GitHub. It should be downloaded from the Github, and the location should be configured in "build_config.py" with "--path_pretrained_emb" option. After following the description in the below link, copy converted "En_vectors.txt" to the same location with "main.py".

link: following the question-5 in https://github.com/nusnlp/nea/blob/master/FAQ.md

For our model and the experiments on the TOEFL, we use the 100-dimensional pre-trained embedding model on Google News, Glove (Pennington, Socher, and Manning 2014). We use pretrained model "XLNet-base" for the variation of XLNet.

Glove link: https://nlp.stanford.edu/projects/glove/

XLNet link: https://github.com/huggingface/transformers/

## Run Models
#### Basic run
A basic run is performed by "main.py" with configuration options by providing in terminal or modifying "build_config.py" file.
Detail information about the configuration can be found in the "build_config.py" ("ilcr_avg" is simple model relying on essay length, and ilcr_kld" is our model incorporating KL-divergence).

	Examples for execution (assume that a data path is given in build_config.py).
    python main.py --essay_prompt_id_train 3 --essay_prompt_id_test 3 --target_model ilcr_kld

#### The list of models
conll17: The automated essay scoring model in Dong et al. (2017)
emnlp18: The coherence model in Mesgar and Strube (2018)
aaai18: The automated essay scoring model in Tay et al. (2018)
ilcr_avg: The baseline which consists of averaged word representations
ilcr_kld: Our model which incorporates KL-divergence to consider essay content

#### Pre-defined configuration
For the convenient reproduction, we provide four configuration examples, a configuration with RNN models (e.g., "asap_build_config.py") and a configuration with XLNet model (e.g., "asap_xlnet_build_config.py") for the two datasets.
The location of the dataset and pretrained embedding layer should be managed properly in "build_config.py".

Note that additional parameters for baseline models should be configured as target models as described in the literatures

## Model Parameters
We describe model parameters as follows, and more details can be found in each configuration files, e.g.,)"build_config.py", "asap_xlnet_build_config.py".

| Dataset  | Learning rate | Droptout | Emb dim | RNN cell size | Batch size | Eps | Activation function |
| ------------- | :---: | :---: | :---: |    :---: |  :---: |  :---: |  :---: |  
| ASAP  | 0.001  | 0.1 | 100 | 150 | 32 | 1e-4 | Leak-Relu |
| TOEFL  | 0.003  | 0.1 | 100 | 150 | 32 | 1e-6 | Leak-Relu |

Note that we apply learning rate as 0.0005 for the prompt 2 in ASAP.

## State of the art in TOEFL
We also compare with the state of the art on TOEFL, Nadeem et al.(2019). We notice that the reported performance in Nadeem et al.(2019) cannot be compared with previous work due to a different experimental setup; they filter out the more than 7.5% of sentences whose length is longer than a length threshold, and they evaluate performance without CV. To ensure a fair comparison, we only modified the experimental setup in their implementation.

We attached both of their original implementation and our modification: "aes_bea19_origin.tar.gz" and "BERT_BCA_train_modified.py" in "aes_bea19_compare.tar.gz", respectively. We only modified their implementation to evaluate performance in 5-fold CV setting, thus it still filters sentences by their length threshold. We observe that the performacne is lower when we do not filter sentences as previous work does. See their Github page for more details.

link: https://github.com/Farahn/AES

## Acknowledge
This implementation was possible thanks to many shared implementations.
