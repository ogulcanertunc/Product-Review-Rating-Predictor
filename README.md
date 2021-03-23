<h1 align='center'>Amazon Electronics Reviews Classifier</h1>

<p align="center">
  <img src="https://images.unsplash.com/photo-1523474253046-8cd2748b5fd2?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80" width=600>
</p>

<strong> Here is a demo application of the review classifier: https://review-rating-predictor-byogi.herokuapp.com// </strong>

Try entering a review you found online or wrote yourself; It will classify the review title, review, or both with a score of 1-5. However, it may be a mistake to expect a very consistent result, because the original data is too large, so I used a scaled-down dataset that I had used in the Rating Systems and Sorting Reviews on Amazon Data project.

</details>

## Structure of Notebooks:
<details>
<a name="Structure"></a>
<summary>Show/Hide</summary>
<br>

1. Pre-Process
   * Convert to lowercase
   * Removing custom formats (html, email, etc.)
   * Remove accented chars
   * Remove_special_chars

2. Models
   * score_funct
   * TF-IDF
   * Train Test Split
   * Decision Tree Classifier
   * Random Forest
   * Logistic Regression
   * Support Vector Machines
   * KNN
   * Adaboost
   * Saving Model
   * Model Evaluation
   * Model Selection


3. Review Application
    * 3.1 Imports
    * 3.2 Data Entry with Streamlit
    * 3.3 Prediction
</details>  

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>NLTK</strong>
* <strong>Scrapy</strong>
* <strong>Scikit-Learn</strong>
* <strong>Spacy</strong>
* <strong>Streamlit</strong>
* <strong>Heroku</strong>
</details>
