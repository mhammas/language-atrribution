# Language Attribution

This project builds a text classification pipeline that attributes user comments to one of four predefined classes (0-3) based on their content. The training data consists of textual descriptions labeled with a class in a CSV file. The system extracts the top 20 TF-IDF keywords per class to create a focused and interpretable feature set. Using this keyword-based vocabulary, both the training descriptions and the target comments (from a JSON file) are vectorized. A Random Forest classifier is trained and evaluated using 10-fold cross-validation to ensure robustness. Finally, the trained model is used to predict the most likely class for each input comment, and the results are saved in a structured JSON output. This pipeline is designed for efficient, interpretable classification of natural language text based on domain-specific patterns.

# Steps

- Load CSV and split description text by class.

- Compute TF-IDF per class subset.

- Select top 20 keywords (by average TF-IDF score) for each class.

- Combine all class keywords into one vocabulary.

- Vectorize descriptions and comments using only those keywords.

- Train model, perform 10-fold CV, and predict.

