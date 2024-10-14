# H&M Personalized Fashion Recommendations Machine Learning Project

![header](https://user-images.githubusercontent.com/60201466/167953233-53ce9848-5da5-481b-a14d-270c794255d1.jpg)

Source: Kaggle [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)

Data source: Kaggle [Data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

## ▶️ Run the project
Please download the data from the Kaggle website and put them in main directory. Then, you can run the training job as follows:
```
pip install -r requirements.txt
python data_processing_and_training.py --model_type <bagging/lgbm/nn> --sample_transactions <true/false>
```
where the model_type is the type of model you want to train and the sample_transactions gives you the option to run the model on a smaller dataset (1/1000 of the original dataset) for testing purposes.


## 📂 Files

👗 articles_EDA.ipynb: EDA for articles dataset

👥 customers_EDA.ipynb: EDA for customers dataset

🧾 transaction_EDA.ipynb: EDA for transactions dataset

🧪 data_combined_EDA.ipynb: EDA for the combined dataset of transactions, customers, and articles

⚙️ data_processing_and_training.py: Feature engineering & model training which is expanded more in the notebook called data_processing_and_training_notebook.ipynb

---



### ✨ Project Overview

Given the purchase history of customers over time, along with supporting metadata, we want to predict what products each customer will purchase in the 7-day period immediately after the training data ends.

## 🌻 Motivation

Online stores offer shoppers an extensive selection of products to browse through, which is often overwhelming. To enhance the shopping experience, product recommendations are key. Not merely to drive sales, but also to reduce returns, and thereby minimizes emissions from transportation.

## 🧠 Model & Result
I first explored the data and extracted insights from it. Then, I combined the data and performed feature engineering on it. I tried three models: Bagging Classifier and Light GBM and Neural Network. The best result was obtained using Light GBM with an accuracy score of 80% and an execution time of 670 ms per test datapoint.

#### Bagging Classifier
- Accuracy score = 78 %
- Execution time = 2.06 s per test datapoint

#### Light GBM
- Accuracy score = 80 %
- Execution time = 670 ms per test datapoint

#### Neural Network
- Accuracy score = 77 %
- Execution time = 5.5 ms per test datapoint



## 🗂 Dataset
We use three datasets in this project:

#### 1. transactions_train.csv
The transaction dataset is the largest database containing everyday transactions in two years period. It contains 31788324 rows × 5 columns:
- t_dat : Date of a transaction in format YYYY-MM-DD but provided as a string.
- customer_id : A unique identifier of every customer (mapped to the customer_id in customers table)
- article_id : A unique identifier of every article (mapped to the article_id in articles table)
- price : Price of purchase
- sales_channel_id : Sales channel 1 or 2

#### 2. customers.csv
Unique indentifier of a customer:
- customer_id : A unique identifier of the customer

5 product related columns:
- FN : Binary feature (1 or NaN)
- Active : Binary feature (1 or NaN)
- club_member_status : Status in a club, 3 unique values
- fashion_news_frequency : Frequency of sending communication to the customer, 4 unique values
- age : Age of the customer
- postal_code : Postal code (anonimized), 352 899 unique values
 
#### 3. articles.csv
Unique indentifier of an article:
- article_id (int64) : A unique 9-digit identifier of the article, 105542 unique values (as the length of the database)

Columns related to the pattern, color, perceived colour (general tone), department, index, section, garment group, and detailed description:
- product_code (int64) : 6-digit product code (the first 6 digits of article_id, 47224 unique values)
- prod_name (object) :Nname of a product, 45875 unique values
- product_type_no (int64) : Product type number, 131 unique values
- product_type_name (object) : Name of a product type, equivalent of product_type_no
- product_group_name (object) : Name of a product group, in total 19 groups
- graphical_appearance_no (int64) : Code of a pattern, 30 unique values
- graphical_appearance_name (object) : Name of a pattern, 30 unique values
- colour_group_code (int64) : Code of a color, 50 unique values
- colour_group_name (object) : Name of a color, 50 unique values
- perceived_colour_value_id : Perceived color id, 8 unique values
- perceived_colour_value_name : Perceived color name, 8 unique values
- perceived_colour_master_id : Perceived master color id, 20 unique values
- perceived_colour_master_name : Perceived master color name, 20 unique values
- department_no : Department number, 299 unique values
- department_name : Department name, 299 unique values
- index_code : Index code, 10 unique values
- index_name : Index name, 10 unique values
- index_group_no : Index group code, 5 unique values
- index_group_name : Index group code, 5 unique values
- section_no : Section number, 56 unique values
- section_name : Section name, 56 unique values
- garment_group_n : Section number, 56 unique values
- garment_group_name : Section name, 56 unique values
- detail_desc : 43404 unique values

