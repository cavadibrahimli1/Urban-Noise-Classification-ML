![image](https://github.com/cavadibrahimli1/Urban-Noise-Classification-ML/assets/76445357/bd40fde3-3b79-426c-8a87-a1cb008a6e0d)


# Urban-Noise-Classification-ML

This project focuses on the classification of urban noise using various machine learning algorithms. The aim is to develop robust models capable of accurately identifying different types of urban sounds, leveraging the UrbanSound8K dataset. The implemented models include Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Random Forest (RF).

## Table of Contents

- [Project Description](#project-description)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Implemented Algorithms](#implemented-algorithms)
- [Methodology](#methodology)
- [Results and Discussion](#results-and-discussion)
- [Challenges](#challenges)
- [Future Directions](#future-directions)
- [System Requirements](#system-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description

Urban noise has grown to be a major problem for both policymakers and city inhabitants as an inevitable outcome of urbanization and industry. This project aims to classify urban noises using machine learning techniques to aid in noise monitoring and management.

## Key Features

- **Dataset**: Utilized the UrbanSound8K dataset, which contains 8,732 labeled sound excerpts from 10 different classes.
- **Machine Learning Models**: Implemented models include DNN, CNN, LSTM, and Random Forest.
- **Feature Extraction**: Used Mel-frequency cepstral coefficients (MFCCs), chroma features, spectral contrast, and more.
- **Evaluation Metrics**: Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset

The UrbanSound8K dataset contains 8,732 labeled sound excerpts from 10 different classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

### Data Preprocessing
- **Normalization**: Ensured all audio files had the same sampling rate and format.
- **Noise Reduction**: Applied techniques to filter out background noise and improve audio quality.
- **Data Splitting**: Divided the dataset into training (70%), validation (15%), and test (15%) sets.

## Implemented Algorithms

### Deep Neural Networks (DNN)
- **Architecture**: Input layer, several hidden layers with ReLU activation, and an output layer with softmax activation.
- **Optimization**: Used the Adam optimizer and categorical cross-entropy loss function.
- **Performance**: Achieved the highest accuracy at 94.5%.

### Convolutional Neural Networks (CNN)
- **Architecture**: Multiple convolutional layers followed by pooling layers, dropout layers for regularization, and dense layers.
- **Feature Extraction**: Effective in capturing spatial hierarchies in audio spectrograms.
- **Performance**: Achieved 90% accuracy.

### Long Short-Term Memory Networks (LSTM)
- **Architecture**: Sequential layers with LSTM units to capture temporal dependencies in audio data.
- **Challenge**: Struggled with capturing long-range dependencies.
- **Performance**: Achieved 79% accuracy.

### Random Forest (RF)
- **Ensemble Learning**: Combines multiple decision trees to reduce overfitting and improve generalization.
- **Feature Importance**: Capable of evaluating the significance of each feature.
- **Performance**: Achieved 87% accuracy.

## Methodology

1. **Data Preprocessing**:
   - Loaded the UrbanSound8K dataset and handled missing or corrupted audio files.
   - Normalized audio levels and applied noise reduction techniques.
   - Split the data into training, validation, and test sets.

2. **Feature Extraction**:
   - Converted audio data into a format suitable for machine learning algorithms.
   - Extracted features such as MFCCs, chroma features, spectral contrast, and more.

3. **Model Building**:
   - Implemented DNN, CNN, LSTM, and RF models with appropriate architectures and hyperparameters.
   - Used libraries like TensorFlow/Keras for neural network models and scikit-learn for Random Forest.

4. **Model Training**:
   - Trained the models using the training dataset and optimized hyperparameters.
   - Monitored validation performance to prevent overfitting.

5. **Model Evaluation**:
   - Evaluated the models using accuracy, precision, recall, F1-score, and confusion matrices to determine performance.

## Results and Discussion

The performance metrics for each model are summarized below:

| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| DNN           | 94.5%    | 95.1%     | 94.0%  | 94.6%    |
| CNN           | 90%      | 91.0%     | 92.1%  | 91.5%    |
| Random Forest | 87%      | 86.5%     | 87.8%  | 87.1%    |
| LSTM          | 79%      | 78.7%     | 79.1%  | 77.4%    |

### Comparative Analysis
- **DNN**: Demonstrated superior performance with the highest accuracy and F1-score. Effective in capturing both low-level and high-level features.
- **CNN**: Performed well with a high accuracy, leveraging convolutional layers to extract spatial features from audio spectrograms.
- **Random Forest**: Showed good generalization capabilities with balanced performance metrics.
- **LSTM**: Highlighted the importance of capturing temporal dependencies, though performance indicated a need for further optimization.

### Implications
The high performance of DNN and CNN models suggests that these architectures are well-suited for urban noise classification, potentially aiding in the development of robust and scalable noise monitoring systems.

## Challenges

1. **Data Quality**: Addressing imbalances and noise in the dataset through data augmentation and transfer learning.
2. **Computational Limits**: Implementing model compression techniques to enable real-time processing on resource-constrained devices.
3. **Model Robustness**: Developing models that can adapt to varying environmental conditions and maintain high performance.

## Future Directions

1. **Automated Sound Event Detection**: Enhance the automation of sound event detection in continuous audio streams.
2. **Feature Optimization**: Improve feature selection and extraction processes for better model efficiency.
3. **Edge Computing**: Explore the deployment of models on edge devices for real-time noise classification.
4. **Community Involvement**: Utilize crowdsourcing for data collection to diversify and expand the dataset.

## System Requirements

- **Processor**: Intel Core i7-13650HX (20 CPUs) @ 2.60GHz
- **Memory**: 16 GB RAM
- **Graphics**: NVIDIA GeForce RTX 4060
- **Operating System**: Windows 11
- **Storage**: 1 TB SSD

## Software Requirements

- **Python Version**: 3.11.9
- **Libraries**: 
  - librosa - 0.10.2.post1
  - numpy - 1.26.4
  - pandas - 2.2.2
  - scikit-learn - 1.4.2
  - seaborn - 0.13.2
  - matplotlib - 3.9.0
  - tensorflow - 2.16.1
  - keras - 3.3.3
  - warnings (part of the Python standard library)
- **IDE**: Visual Studio Code

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Urban-Noise-Classification-ML.git
    cd Urban-Noise-Classification-ML
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## Usage

- Open the provided Jupyter Notebook (`ehb328eUrbanSound.ipynb`) to explore the implementation details.
- Follow the steps in the notebook to preprocess the data, extract features, train the models, and evaluate their performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The UrbanSound8K dataset creators
- Istanbul Technical University, Electronics and Communications Engineering Department

