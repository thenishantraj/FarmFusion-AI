
# FarmFusion-AI - Crop and Fertilizer Recommendation System

## Project Overview
FarmFusion AI is a machine learning-based recommendation system that suggests the best crops to grow and the most suitable fertilizers based on soil conditions and environmental factors. This project aims to help farmers make data-driven decisions to optimize yield and improve agricultural sustainability.

## Features
- Predict the best crop to cultivate based on soil properties.
- Recommend suitable fertilizers based on crop and soil conditions.
- Data visualization for better insights.
- User-friendly interface for predictions.

## Dataset
The project uses the **Crop Recommendation Dataset**, which contains information on soil nutrients (NPK values), temperature, humidity, and pH levels, along with the best-suited crop for each set of conditions.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Machine Learning Algorithm:** Classification models (Decision Trees, Random Forest, SVM, etc.)
- **Jupyter Notebook** for development and analysis

## Installation
To run the project, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/farmfusion-ai.git
   cd farmfusion-ai
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Crop_Fertilizer.ipynb
   ```
3. Run the cells step by step to load the dataset, preprocess the data, train the model, and make predictions.

## Model Training
The dataset undergoes preprocessing, feature selection, and splitting into training and testing sets. Machine learning models such as Decision Trees and Random Forest are trained and evaluated for accuracy.

## Results
- **Model Accuracy:** Evaluated using metrics like accuracy, precision, recall, and F1-score.
- **Visualization:** Heatmaps, bar charts, and scatter plots are used for data analysis.

## Future Improvements
- Deploy as a web or mobile application.
- Integrate real-time weather data for more accurate recommendations.
- Improve model accuracy using deep learning techniques.

## Contributors
- **Your Name** - Developer
- **Other Contributors** - [Add names if applicable]

## License
This project is open-source and available under the [MIT License](LICENSE).
```

