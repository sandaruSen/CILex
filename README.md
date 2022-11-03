# CILex

![alt text](https://github.com/sandaruSen/cilex/blob/main/figs/architecture.png?raw=true)

["CILex: An Investigation of Context Information for Lexical Substitution Methods "](https://aclanthology.org/2022.coling-1.362.pdf), 
In Proceedings of the 29th International Conference on Computational Linguistics, 2022

Sandaru Seneviratne, Elena Daskalaki, Artem Lenskiy, Hanna Suominen 


### How to run the scripts
1.  Install requirements
    ```shell script
    pip install -r requirements.txt
    ```
2. Go to configs/subst_generators/lexsub/ (You can find the scripts for cilex1,cilex2, and cilex3 in there)
3. Correctly define the paths of the files. 
4. Run the required file.

## Results
Results of the substitution generation are presented in the following table. 
![alt text](https://github.com/sandaruSen/cilex/blob/main/figs/results.png?raw=true)

Results of the candidate ranking are presented in the following table. 
![alt text](https://github.com/sandaruSen/cilex/blob/main/figs/gap.png?raw=true)

Results of the ablation study are presented in the following table. 
![alt text](https://github.com/sandaruSen/cilex/blob/main/figs/ablation.png?raw=true)


### Citation
If you use our work, please cite our paper:



```
@inproceedings{seneviratne2022cilex,
  title={CILex: An Investigation of Context Information for Lexical Substitution Methods},
  author={Seneviratne, Sandaru and Daskalaki, Elena and Lenskiy, Artem and Suominen, Hanna},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={4124--4135},
  year={2022}
}
```


## Acknowledgement

The implementation of CILex relies on resources from [XLNet+embs](https://github.com/Samsung/LexSubGen/tree/main/lexsubgen)
and [LexSubCon](https://github.com/gmichalo/LexSubCon/) and we thank the authors for their support. 

The code is implemented using [PyTorch](https://github.com/pytorch/pytorch).







