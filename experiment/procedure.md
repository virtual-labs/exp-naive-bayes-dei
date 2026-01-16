The objective of this experiment is to classify the text messages into two categories—Spam and Ham. The input to the model consists of multiple independent random text messages, while the output is a binary dependent variable indicating the category of that message.

**Step 1**: **Import Libraries**: Initialize the environment by importing the required libraries: `pandas`, `numpy`, `re`, `matplotlib`, `nltk`, and the necessary modules from `scikit-learn`.

**Step 2**: **Load Dataset**: Load the `Spam_Detection.csv` file into a DataFrame using `pd.read_csv()`. The dataset contains 1742 text messages, where each row represents an individual message and its corresponding category (Spam or Ham).

**Step 3**: **Exploratory Data Analysis**: Conduct initial data exploration using `head()`, `tail()`, `info()`, `shape`, and `value_counts()` to understand the dataset structure and class distribution.

**Step 4**: **Check Missing Values**: Inspect the dataset for any missing entries using `isnull().sum()` to ensure data quality before processing.

**Step 5**: **Visualize Class Distribution**: Create visual representations, such as pie charts, to illustrate the balance between Spam and Ham messages in the dataset.

**Step 6**: **Initialize Preprocessing Tools**: Set up essential Natural Language Processing (NLP) tools, including the `PorterStemmer` and the standard English stop-word list.

**Step 7**: **Text Preprocessing**: Clean the text data by implementing functions to convert text to lowercase, remove URLs, strip special characters and numbers, and eliminate redundant white spaces.

**Step 8**: **Target Encoding**: Transform the categorical labels into numerical format, where **Ham** is encoded as **0** and **Spam** is encoded as **1**.

**Step 9**: **Data Partitioning**: Split the preprocessed dataset into training (80%) and testing (20%) sets to facilitate model building and evaluation.

**Step 10**: **TF-IDF Vectorization**: Convert the textual messages into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique.

**Step 11**: **Model Training**: Initialize and train a **Multinomial Naive Bayes** model using the vectorized training data.

**Step 12**: **Performance Evaluation**: Assess the model's effectiveness using metrics such as Accuracy Score, Precision, Recall, and F1-score, supplemented by a Confusion Matrix and Precision-Recall curve visualizations.

**Step 13**: **Prediction Analysis**: Evaluate the model's real-world applicability by analyzing predictions on random test samples, displaying the actual label, predicted label, and the calculated spam probability for each sample.