## Interval-Valued Fuzzy and Intuitionistic Fuzzy K-Nearest Neighbor (IVF-IFKNN) for Imbalanced Data Classification  

This repository contains the implementation of the methods presented in the paper:  
**[Interval–valued fuzzy and intuitionistic fuzzy–KNN for imbalanced data classification](https://www.sciencedirect.com/science/article/pii/S0957417421009209)** by Saeed Zeraatkar and [Fatemeh Afsari](f.afsari@ufl.edu).  

Published in *Expert Systems with Applications*, this paper introduces advanced fuzzy-based approaches for effectively classifying imbalanced datasets, leveraging interval-valued and intuitionistic fuzzy K-Nearest Neighbor (IVF-IFKNN) methods.  

---

## Abstract  

The classification of imbalanced datasets is a prevalent challenge in machine learning. Traditional methods often fail to correctly classify minority classes, resulting in suboptimal performance. This paper proposes novel adaptations of K-Nearest Neighbor (KNN) using interval-valued fuzzy sets and intuitionistic fuzzy logic, designed to handle imbalanced class distributions while maintaining robustness and interpretability. The proposed approaches are validated through extensive experiments on synthetic and real-world datasets.  

---

## Features  

- **Interval-Valued Fuzzy KNN (IVF-KNN):** Enhances the traditional KNN algorithm with interval-valued fuzzy logic to improve classification performance in imbalanced datasets.  
- **Intuitionistic Fuzzy KNN (IF-KNN):** Incorporates intuitionistic fuzzy sets to manage uncertainty and imprecision effectively.  
- **Imbalance Handling:** Tailored to mitigate the effects of class imbalance, improving recall and F1 scores for minority classes.  

---

## Repository Structure  

- **`src/`**: Contains the main implementation of IVF-KNN and IF-KNN methods.  
- **`data/`**: Example datasets used for testing and evaluation.
- **`results/`**: Output files and performance metrics from experiments.  
- **`README.md`**: This file.  

---

## Installation  

To run the code, ensure you have the following installed:  

- Python 3.8 or later  
- Required Python libraries (install via `requirements.txt`):  
  ```bash  
  pip install -r requirements.txt  
  ```  

---

## Usage  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/fafsari/IVI_FuzzyKNN-Imbalanced.git  
   cd IVI_FuzzyKNN-Imbalanced 
   ```  

2. **Prepare Your Dataset**  
   Add your dataset in the `data/` directory. Ensure it is in a compatible format (e.g., CSV).  

3. **Run the Algorithm**  
   ```bash  
   python src/run_main.py   
   ``` 

---

## Results  

The proposed IVF-IFKNN methods demonstrate superior performance over traditional KNN on benchmark datasets, particularly in handling class imbalance. For detailed results, refer to the `results/` directory or the published paper.  

---

## Citation  

If you use this repository in your research, please cite the original paper:  

```bibtex  
@article{saeed2021interval,
  title={Interval-valued fuzzy and intuitionistic fuzzy-KNN for imbalanced data classification [J]},
  author={Saeed, Zeraatkar and Fatemeh, Afsari},
  journal={Expert Systems With Applications},
  volume={2021},
  number={184},
  year={2021}
}  
```  

---

## Acknowledgments  

This repository is maintained by the corresponding author of the paper. Special thanks to the open-source community for providing tools and libraries that supported the implementation and experiments.  

---

## License  

This code is licensed under the [MIT License](LICENSE).  

---  
