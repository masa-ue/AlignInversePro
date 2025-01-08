# Tutorial: Inference-Time Alignment in Discrete Diffusion Models for Protein Design 

This code accompanies the tutorial paper on inference-time alignment in diffusion models. Here, the goal is to optimize several reward functions in a protein inverse folding model ($p(x|c)$) where $x$ is a sequence, and $c$ is a backbone structure. 

As propsoed in [Cambell and Yim et.al 2024]() and recent related works such as Evodiff and , a discrete diffusion model is used as a representative model for protein design that characetrized a natural protein-like space. Beyond these works, in our tutorial, we have exaplined how to optimize several downstream reward functions while keeping its ``naturalness''. In this repository, we see how it practically works in the context of protein design. 


## How to Run 
Go to `./fmif` folder. Then, the inference-time technique can be run as follows.  

```bash 
CUDA_VISIBLE_DEVICES=4 python eval_finetune.py --decoding 'original' --reward_name 'scRMSD'  --repeatnum 20
```

* **--decoding**: 
  * **SMC**: Refer to Sec. 3.1 or papers . 
  * **SVDD** (a.k.a. value-based sampling): Sec. 3.2 or the paper 
  * **NestedIS**: Refert to Sec. 3.3
  * **Classifier guidance**: Refer to Sec. 5.2  or the paper such as  
* **--rewards**:  
  * **stability**: This is a reward trained in [Wang and Uehara et al., 2024](https://arxiv.org/abs/2410.13643), which predicts Gibbs’s free energy from a sequence and a structure on the [Megalscale](https://www.nature.com/articles/s41586-023-06328-6). For details, refer to the [code](https://github.com/ChenyuWang-Monica/DRAKES).  
  * **pLDDT** (non-differentiable): Common metric to characterize the confidecen of prediction. It has been used as a certain proxy of stability. 
  * **scRMSD** (non-differentiable): $\|c- f(\hat x) \|$ where $f$ is a forward folding model ([ESMfold](https://github.com/facebookresearch/esm)). While the pre-trained model is already a conditoinal diffusoin model, this is considered to be usesful to robustify the generated protein further. 
  * **stability_rosetta** (non-differentiable): $g(f(\hat x))$ where $f$ is a forward folding model ([ESMfold](https://github.com/facebookresearch/esm)) and $g$ is physics-based reward feedback to calcuate energy after packing ([Pyrosetta](https://www.pyrosetta.org/)). This task has been considered in 
*  **--repeat_num**: When using SMC, SVDD, Nested IS, we need to choose the dupilicatoin hyperparaeter. 
* **--alpha**: We set this as $0.5$ in SMC and classfier guidance by default. For SVDD, we choose $0.0$ by default. 


## Outputs  

We condition on several wild backbone strucres in vadliation protein datasets. We save each generated protein as a pdb fild in the folder `./sc_tmp/`. We also record several important statistics as a pandas format in the folder `./log`. 

## Results 

Each blue point correponds to the median RMSD of generated samples for each backbone structure. For example, when optimizng scRMSD, for some protein, while naive inference procedures have certian incosisntecy, inference-time technique can make the generated result very consistent with the forward folding model.  

![image](media.jpeg)

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
@article{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2402.04997},
  year={2024}
}
```

