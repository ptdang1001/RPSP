# Generalized Matrix Local Low Rank Representation by Random Projection and Submatrix Propagation

This repository contains the implementation of the paper: **"Generalized Matrix Local Low Rank Representation by Random Projection and Submatrix Propagation"**.

## Table of Contents
- Introduction
- Requirements
- Installation
- Usage
- Citation
- Contributing
- License
- Contact

## Introduction
This project implements the algorithm described in the paper "Generalized Matrix Local Low Rank Representation by Random Projection and Submatrix Propagation". The goal is to provide an efficient and scalable method for matrix decomposition and representation.

## Requirements
The required dependencies are listed in the `requirements.yml` file. You can create a conda environment with these dependencies.

## Installation
To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ptdang1001/RPSP.git
    cd RPSP
    ```

2. Create a conda environment using the `requirements.yml` file:
    ```bash
    conda env create -f requirements.yml
    ```

3. Activate the environment:
    ```bash
    conda activate rpsp
    ```

## Usage
To run the code, use the following command:
```bash
python main.py
```

Make sure to customize any parameters or configurations in the `main.py` file as needed for your specific use case.

## Citation
If you use this code in your research, please cite the following paper:
```
@inproceedings{dang2023generalized,
  title={Generalized Matrix Local Low Rank Representation by Random Projection and Submatrix Propagation},
  author={Dang, Pengtao and Zhu, Haiqi and Guo, Tingbo and Wan, Changlin and Zhao, Tong and Salama, Paul and Wang, Yijie and Cao, Sha and Zhang, Chi},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={390--401},
  year={2023}
}
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or issues, please contact zhangchi@ohsu.edu/dangpe@ohsu.edu
