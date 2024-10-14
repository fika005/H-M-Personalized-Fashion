import pdb
import fire
import pandas as pd
import numpy as np
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def load_and_preprocess_article_data(file_path):
    """
    Load and preprocess article data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing article data.

    Returns:
        pd.DataFrame: Preprocessed article data.
    """
    df_article = pd.read_csv(file_path)
    df_art = df_article.copy()
    
    
    columns_to_drop = ['product_code', 'prod_name', 'product_type_no',
                       'graphical_appearance_no', 'colour_group_code', 'colour_group_name',
                       'perceived_colour_value_id', 'perceived_colour_master_id', 'perceived_colour_master_name',
                       'department_no', 'department_name', 'index_code', 'index_name',
                       'index_group_no', 'section_no', 'garment_group_no', 'detail_desc']
    
    df_art.drop(columns_to_drop, inplace=True, axis=1)
    
    def modify_column(df, column_name, threshold):
        temp = df[column_name].value_counts() <= threshold
        temp = temp.loc[temp]
        mask = set(temp.index)
        df[f"modified_{column_name}"] = df[column_name].apply(lambda x: 'other' if x in mask else x)
    
    modify_column(df_art, 'graphical_appearance_name', 2000)
    modify_column(df_art, 'product_group_name', 2000)
    modify_column(df_art, 'section_name', 3328)
    modify_column(df_art, 'garment_group_name', 4874)
    temp = df_art['product_type_name'].value_counts() <= 1320
    temp = temp.loc[temp]
    mask = set(temp.index)
    df_art["modified_product_type"] = df_art["product_type_name"].apply(lambda x: 'other' if x in mask else x)
    
    columns_to_drop = ['graphical_appearance_name', 'product_group_name', 'section_name', 'garment_group_name', 'product_type_name']
    df_art.drop(columns_to_drop, inplace=True, axis=1)
    
    return df_art

def load_and_preprocess_customer_data(file_path):
    """
    Load and preprocess customer data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing customer data.

    Returns:
        pd.DataFrame: Preprocessed customer data.
    """
    df_customer = pd.read_csv(file_path)
    df_cust = df_customer.copy()
    
    df_cust.drop(['FN', 'fashion_news_frequency', 'postal_code'], inplace=True, axis=1)
    df_cust['club_member_status'] = df_cust['club_member_status'].fillna('N/A')
    df_cust['Active'] = df_cust['Active'].fillna(0)
    df_cust["age"] = df_cust["age"].fillna(df_cust["age"].mean())
    
    df_cust = pd.concat([df_cust.drop(columns=["club_member_status"]), pd.get_dummies(df_cust["club_member_status"])], axis=1)
    
    return df_cust

def load_and_preprocess_transaction_data(file_path, df_art, sample_transactions):
    """
    Load and preprocess transaction data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing transaction data.
        df_art (pd.DataFrame): Preprocessed article data.

    Returns:
        pd.DataFrame: Preprocessed transaction data merged with article data.
    """
    df_transaction = pd.read_csv(file_path)
    if sample_transactions:
        df_transaction = df_transaction.sample(frac=0.001)
    df_transaction = df_transaction.sort_values("customer_id")
    df_transaction["t_dat"] = pd.to_datetime(df_transaction["t_dat"])
    
    df_combined = df_transaction.merge(df_art, on=["article_id"], how="inner")
    
    random_time = df_combined[df_combined.t_dat < "2020-09-15"].groupby("customer_id").agg({"t_dat": lambda x: x.sample(1)})
    random_time = random_time.rename(columns={"t_dat": "pivot"}).reset_index()
    
    df_combined_random = df_combined.merge(random_time, on="customer_id")
    df_combined_random.sales_channel_id = df_combined_random.sales_channel_id.astype(str)
    
    return df_combined_random

def process_data(df, df_article, all_article_ids, df_cust, feature_columns):
    """
    Process the combined data to create features for model training.

    Args:
        df (pd.DataFrame): Combined transaction and article data.
        df_article (pd.DataFrame): Preprocessed article data.
        all_article_ids (list): List of all unique article IDs.
        feature_columns (list): List of column names to use as features.

    Returns:
        pd.DataFrame: Processed data ready for model training.
    """
    dummy_cols = []
    for col in feature_columns:
        dummy_cols.extend([col + "_" + x for x in df[col].unique()])
    
    def process_group(df):
        before = df[df["t_dat"] <= df["pivot"]]
        before_7d = df[(df["t_dat"] <= df["pivot"]) & (df["t_dat"] > df["pivot"] - pd.Timedelta(7, "days"))]
        after = df[(df["t_dat"] > df["pivot"]) & (df["t_dat"] < df["pivot"] + pd.Timedelta(7, "days"))]
        after["label"] = 1
        after = after[["article_id", "label"]]
        if len(after) == 0:
            return None
        negs = random.sample(all_article_ids, len(after))
        negs = [i for i in negs if i not in df.article_id]
        negs_df = after.copy()
        negs_df["article_id"] = negs
        negs_df["label"] = 0
        after = pd.concat([after, negs_df], axis=0)
        means = pd.get_dummies(before[feature_columns]).mean(axis=0)
        means["avg_price"] = before["price"].mean()
        means_df = means.to_frame().transpose()
        means_df = means_df.reindex(columns=dummy_cols).fillna(0.0)
        means_df["num_purchase"] = len(before)
        means_7d = pd.get_dummies(before_7d[feature_columns]).mean(axis=0)
        means_df_7d = means_7d.to_frame().transpose()
        means_df_7d.columns = means_df_7d.columns + "_7d"
        means_df_7d = means_df_7d.reindex(columns=[col + "_7d" for col in dummy_cols]).fillna(0.0)
        means_df_7d["num_purchase_7d"] = len(before_7d)
        means_df_7d["avg_price_7d"] = before_7d["price"].mean()
        means_df_combined = pd.concat([means_df, means_df_7d.fillna(0)], axis=1)
        combined = pd.concat([means_df_combined] * len(after), axis=0)
        combined["article_id"] = after["article_id"].values
        combined["label"] = after["label"].values
        return combined.reset_index(drop=True)
    
    final_data = df.groupby("customer_id").apply(process_group).reset_index(level=1, drop=True).reset_index()
    
    one_hot_article = pd.get_dummies(df_article[feature_columns[1:]])
    one_hot_article["article_id"] = df_article["article_id"]
    one_hot_article.columns = one_hot_article.columns + "_target"
    
    final_data_ = final_data.merge(df_cust, on="customer_id").merge(one_hot_article, left_on="article_id", right_on="article_id_target")
    final_data_ = final_data_.drop(columns=["article_id_target", "article_id"])
    
    final_data_ = final_data_.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    return final_data_

class Model(pl.LightningModule):
    """
    PyTorch Lightning module for the neural network model.
    """
    def __init__(self, customer_dim, article_dim, hidden_dim):
        super(Model, self).__init__()
        self.customer_layer1 = torch.nn.Linear(customer_dim, hidden_dim)
        self.customer_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.article_layer1 = torch.nn.Linear(article_dim, hidden_dim)
        self.article_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, article_features):
        x = self.customer_layer2(F.relu(self.customer_layer1(x)))
        y = self.article_layer2(F.relu(self.article_layer1(article_features)))
        return (x * y).sum(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, z, y = train_batch
        loss = F.binary_cross_entropy_with_logits(self.forward(x, z), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, z, y = val_batch
        loss = F.binary_cross_entropy_with_logits(self.forward(x, z), y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

class Dataset(Dataset):
    """
    Custom Dataset class for PyTorch DataLoader.
    """
    def __init__(self, features, labels, num_customer_features):
        self.customer_features = torch.tensor(features[:, :num_customer_features], dtype=torch.float32)
        self.article_features = torch.tensor(features[:, num_customer_features:], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.customer_features)
    
    def __getitem__(self, i):
        return self.customer_features[i, :], self.article_features[i, :], self.labels[i]

def train_model(X_tr, y_tr, X_val, y_val, model_type='nn'):
    """
    Train the selected model and report training and validation metrics.

    Args:
        X_tr (pd.DataFrame): Training features.
        y_tr (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        model_type (str): Type of model to train ('nn', 'lgbm', or 'bagging').

    Returns:
        object: Trained model.
    """
    if model_type == 'nn':
        train_set = Dataset(X_tr.values, y_tr.values, 151)
        val_set = Dataset(X_val.values, y_val.values, 151)
        train_loader = DataLoader(train_set, batch_size=32, num_workers=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, num_workers=16)

        model = Model(151, 69, 50)
        trainer = pl.Trainer(max_epochs=4)
        trainer.fit(model, train_loader, val_loader)
        
        train_preds = model(torch.tensor(X_tr.values[:, :151], dtype=torch.float32), 
                            torch.tensor(X_tr.values[:, 151:], dtype=torch.float32)) > 0
        val_preds = model(torch.tensor(X_val.values[:, :151], dtype=torch.float32), 
                          torch.tensor(X_val.values[:, 151:], dtype=torch.float32)) > 0
        
        print(f"Neural Network - Train Accuracy: {accuracy_score(y_tr, train_preds)}")
        print(f"Neural Network - Validation Accuracy: {accuracy_score(y_val, val_preds)}")
        
    elif model_type == 'lgbm':
        model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, n_estimators=1000, num_leaves=70)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val), (X_tr, y_tr)],
                  eval_metric=["binary_error"])
        
        train_preds = model.predict(X_tr)
        val_preds = model.predict(X_val)
        
        print(f"LightGBM - Train Accuracy: {accuracy_score(y_tr, train_preds)}")
        print(f"LightGBM - Validation Accuracy: {accuracy_score(y_val, val_preds)}")
        
    elif model_type == 'bagging':
        model = BaggingClassifier(DecisionTreeClassifier())
        model.fit(X_tr, y_tr)
        
        train_preds = model.predict(X_tr)
        val_preds = model.predict(X_val)
        
        print(f"Bagging Classifier - Train Accuracy: {accuracy_score(y_tr, train_preds)}")
        print(f"Bagging Classifier - Validation Accuracy: {accuracy_score(y_val, val_preds)}")
    
    else:
        raise ValueError("Invalid model type. Choose 'nn', 'lgbm', or 'bagging'.")
    
    return model


def main(model_type='nn', sample_transactions=False):
    print("Starting data loading and preprocessing...")
    df_art = load_and_preprocess_article_data('articles.csv')
    df_cust = load_and_preprocess_customer_data('customers.csv')
    df_combined_random = load_and_preprocess_transaction_data('transactions_train.csv', df_art, sample_transactions)
    print("Data loading and preprocessing completed.")
    
    print("Defining feature columns...")
    feature_columns = ["sales_channel_id", "perceived_colour_value_name",
                       "index_group_name", "modified_graphical_appearance_name",
                       "modified_product_group_name", "modified_section_name", "modified_garment_group_name",
                       "modified_product_type"]
    print("Feature columns defined.")
    
    print("Extracting unique article IDs...")
    all_article_ids = list(df_art.article_id.unique())
    print("Unique article IDs extracted.")
    
    print("Processing data...")
    final_data_ = process_data(df_combined_random, df_art, all_article_ids, df_cust, feature_columns)
    print("Data processing completed.")
    
    print("Preparing features and target for model training...")
    X = final_data_.drop(columns=["label", "customer_id"])
    y = final_data_["label"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    print("Features and target prepared.")
    
    print("Training model...")
    model = train_model(X_tr, y_tr, X_val, y_val, model_type)
    print("Model training completed.")
    

if __name__ == "__main__":
    fire.Fire(main)