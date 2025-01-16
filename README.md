# Tutorial: Inference-Time Alignment in Discrete Diffusion Models for Protein Design 

This code is provided alongside the tutorial paper on inference-time alignment in diffusion models. The objective is to optimize multiple reward functions within a protein inverse folding model ($p(x|c)$), where $x$ represents a sequence, and $c$ denotes a backbone structure.

We employ an inverse folding model (mapping backbone structure to sequence) based on a discrete diffusion model as the foundational model. In this repository, we detail the process of optimizing various downstream reward functions in this diffusion model using inference-time techniques.

## How to Run 
Go to `./fmif` folder. Then, the inference-time technique can be run as follows.  

```bash 
CUDA_VISIBLE_DEVICES=7 python eval_finetune.py --decoding 'SVDD' --reward_name 'LDDT'  --repeatnum 10 --batchsize 5
```

``` 
CUDA_VISIBLE_DEVICES=5 python eval_finetune.py --decoding 'DDBFS' --reward_name 'LDDT'  --repeatnum 5 --batchsize 5 --wandb_name w5d3-expo-expo2
```

* **--decoding**: 
  * **SMC**: Refer to Sec. 3.1 or papers . 
  * **SVDD** (a.k.a. value-based sampling): Sec. 3.2 or the paper 
  * **NestedIS**: Refert to Sec. 3.3
  * **Classifier guidance**: Refer to Sec. 5.2  or the paper such as  
* **--rewards**:  
  * **stability**: This is a reward function trained in [Wang and Uehara et al., 2024](https://arxiv.org/abs/2410.13643), which predicts Gibbs’s free energy from a sequence and a structure on the [Megalscale dataset](https://www.nature.com/articles/s41586-023-06328-6). For details, refer to the [code](https://github.com/ChenyuWang-Monica/DRAKES).  
  * **LDDT**: Common metric to characterize the confidecen of prediction. It has been used as a certain proxy of stability. 
  * **scRMSD**: $\|c- f(\hat x) \|$ where $f$ is a forward folding model ([ESMfold](https://github.com/facebookresearch/esm)). While the pre-trained model is already a conditoinal diffusoin model, this is considered to be usesful to robustify the generated protein further. 
*  **--repeat_num**: When using SMC, SVDD, Nested IS, we need to choose the dupilicatoin hyperparaeter.
* **--batchsize**: Batch size  
* **--alpha**: We set this as $0.5$ in SMC and classfier guidance by default. For SVDD, we choose $0.0$ by default. 


## Outputs  

We condition on several wild backbone strucres in vadliation protein datasets. We save each generated protein as a pdb fild in the folder `./sc_tmp/`. We also record several important statistics as a pandas format in the folder `./log`. 

## Results 

Each blue point correponds to the median RMSD of generated samples for each backbone structure. For example, when optimizng scRMSD, for some protein, while naive inference procedures have certian incosisntecy, inference-time technique can make the generated result very consistent with the forward folding model.  

![image](./media/media.jpeg)

## Installation 

* The pre-trained model is based on the code in multiflow code [Campbell & Yim et al., 2024](https://github.com/jasonkyuyim/multiflow). 
* Then, to introduce weights on pre-trained models, run 
```bash 
python download_model_data.py
```
Then, the dataset will be placed on the folder `./datasets`
* To calculate the energy, we need to install [Pyrosseta](https://www.pyrosetta.org/). 

## Citation 

If you use this codebase, then please cite
```
XXX
```

