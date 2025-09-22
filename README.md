# Official Repository of the paper _Logit Adjusted Test-Time Adaptaion under Partial Class Imbalance_


This code is created on top of the open source repository https://github.com/mariodoebler/test-time-adaptation.

## Important Files
```
- classification/
    -methods
        -experiments
            -tent_la_apx.py
            -eata_la_apx.py
            -sar_la_apx.py
            -deyo_la_apx.py
    
    -run_scripts
        -imagenet_c.sh
        -cifar_100_c.sh
        -domainnet126.sh
```

## Setting Up Datasets

This paper includes experiments for the datasets CIFAR-100-C, ImageNet-C and DomainNet-126. Before running the experiments, make sure you place the datasets inside your data directory `data_dir` as follows with the same names.

```
-data_dir
    -IMAGENET
    -IMAGENET-C
    -CIFAR-100-C
    -cifar-100-python
    -DomainNet-126
```

Here `cifar-100-python`, `IMAGENET` refer to the original, clean CIFAR-100, ImageNet datasets. They will be utilized for experiments related to EATA algorithm.

## Setting up the Environment
This repository requires installing the dependancies in the following `env.yaml` file into an Anaconda environment.

## Running an Example

This example simulates our logit adjustment method for `TENT` algorithm on CIFAR-100-C dataset.

1. Change the following line inside the ```run_scripts/cifar_100_c.sh``` into your desired `data_dir` location.
```
data_dir="<path_to_data_dir>"
```
2. activate the conda environment `tta`
3. Change the current working directory to `<repository>/classification`
4. Execute the following command
```
bash run_scripts/cifar_100_c.sh
```

## Changing Configurations 
The variables listed below can be changed in the following way

* `severity` should bet `3` or `5`
* `algorithm` should be one of `tent`, `eata`, `sar` or `deyo`.
* `partial_classes` should be the subset of class labels in the target dataset. Example classes are provided in the run scripts which corresponds to CIR=0.1 (Class Inclusion Ratio according to the paper)


