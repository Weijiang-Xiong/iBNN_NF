## Normalizing Flows for Implicit BNN

This repo contains the files related to Weijiang Xiong's summer project at PML group. This project aims to learn a more flexible posterior for the input uncertainty with normalizing flow. 

Please look below for the directory structure 

```bash
PML_Internship
    |___code_refs # opensource projects that provided help for this project
    			|___ibnn_code # the code for the base work ibnn
    |___codes # codes for this project
    			|___flows # implemented flows
    			|___models # deterministic and stochastic models
    			|___utils # utility and helper functions
    			|___mnist_lenet.ipynb # notebook to train lenet on FashionMNIST
    			|___nf_example.ipynb # analyze the effect of normalizing flow with logistic regression and MLP 
    			|___result_analysis.ipynb # notebook to visualize results 
    			|___test_lenet.py # script to train lenet on FashionMNIST
    			|___test_vgg.py  # script to train VGG on CIFAR10
    |___HPC course #  tutorial for Triton Cluster at Aalto
    |___project_summary # summary report for this project
    |___references # nice papers about bayesian deep learning and normalizing flow 
    |___updates # monthly update and work presentation slides
```



### Project Setup 

The code mainly depends on PyTorch, any proper installation should be ok. 

The notebook `result_analysis.ipynb` needs trained VGG16 to visualize the prediction results, and the models could be downloaded [here](https://drive.google.com/drive/folders/1LA6HjASTum4DXPZFbFaDZ6PClFQ99X4r?usp=sharing). Please place them under `models/trained/` or modify the path in the notebook. 

