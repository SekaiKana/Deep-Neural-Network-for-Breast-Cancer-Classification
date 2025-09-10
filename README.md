# Deep-Neural-Network-for-Breast-Cancer-Classification

# Deep Neural Network for Breast Cancer Classification

A PyTorch implementation of a deep neural network for binary classification of breast cancer cells using the Wisconsin Diagnostic Breast Cancer Dataset.

## Project Overview

This project demonstrates the implementation of a deep learning model for medical diagnosis, specifically classifying breast cancer cells as benign or malignant. The model achieves strong performance through careful data preprocessing, balanced sampling, and neural network architecture design.

## Features

- **Data Preprocessing**: Automated data loading, cleaning, and standardization
- **Balanced Dataset**: Handles class imbalance by sampling equal numbers from each class
- **Deep Learning Architecture**: Custom PyTorch neural network with configurable layers
- **Training Pipeline**: Complete training loop with loss tracking and visualization
- **Model Comparison**: Includes Keras implementation for performance benchmarking
- **Visualization**: Training and validation loss curves for model evaluation

## Dataset

The project uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) which contains:
- **569 samples** of breast cancer cell measurements
- **30 features** describing cell nucleus characteristics
- **2 classes**: Malignant (M) and Benign (B)

## Model Architecture

### PyTorch Implementation
```
Input Layer (30 features)
    ↓
Hidden Layer (64 neurons, ReLU activation)
    ↓
Output Layer (2 classes, Softmax)
```

### Key Components
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 2
- **Epochs**: 12
- **Data Split**: 80% training, 20% testing

## Results

The model demonstrates effective learning with:
- Decreasing training loss over epochs
- Stable validation performance
- Comparable results to Keras implementation
- No signs of overfitting

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook to execute the complete pipeline:

```bash
jupyter notebook "Data Augmentation.ipynb"
```

The notebook includes:
1. Data loading and exploration
2. Preprocessing and balancing
3. Model definition and training
4. Performance visualization
5. Keras comparison implementation

## File Structure

```
breast-cancer-classification/
│
├── Data Augmentation.ipynb    # Main notebook with complete implementation
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── results/                   # Generated plots and model outputs
```

## Technical Skills Demonstrated

- **Deep Learning**: PyTorch neural network implementation
- **Data Science**: Data preprocessing, visualization, and analysis
- **Machine Learning**: Classification, model evaluation, cross-validation
- **Python Libraries**: PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn
- **Medical AI**: Healthcare data handling and binary classification

## Key Learning Outcomes

1. **Neural Network Design**: Understanding layer architecture and activation functions
2. **Data Preprocessing**: Standardization, train-test splitting, and class balancing
3. **Training Optimization**: Loss function selection and hyperparameter tuning
4. **Model Evaluation**: Loss visualization and performance comparison
5. **Framework Comparison**: PyTorch vs. Keras implementation differences

## Future Enhancements

- [ ] Implement cross-validation for robust performance evaluation
- [ ] Add regularization techniques (dropout, L2 regularization)
- [ ] Experiment with different architectures (deeper networks, different activations)
- [ ] Include additional evaluation metrics (precision, recall, F1-score)
- [ ] Deploy model as a web application

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the breast cancer dataset
- PyTorch and Keras communities for excellent documentation
- Medical research community for advancing cancer diagnosis through AI

## Contact

Feel free to reach out for questions or collaboration opportunities:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)
