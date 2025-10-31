
"""
PESTLE Classification System - Enhanced with Apache Spark, SQLite & Analytics
=============================================================================
Production-ready web interface with distributed processing, data persistence,
and comprehensive insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, StringIndexer, IndexToString
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression as SparkLR
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class DatabaseManager:
    """Manage SQLite database for predictions and model metadata"""
    
    def __init__(self, db_path="pestle_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT
            )
        """)
        
        # Model metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                f1_score REAL,
                precision_score REAL,
                recall_score REAL,
                trained_date DATETIME,
                training_time REAL,
                n_samples INTEGER,
                n_train INTEGER,
                n_test INTEGER,
                framework TEXT,
                is_active BOOLEAN DEFAULT 0
            )
        """)
        
        # Training history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                training_time REAL,
                n_samples INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Category metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                category TEXT NOT NULL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                support INTEGER,
                FOREIGN KEY (model_id) REFERENCES model_metadata(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, text, category, confidence, model_version):
        """Save a prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (text, predicted_category, confidence, model_version)
            VALUES (?, ?, ?, ?)
        """, (text[:500], category, confidence, model_version))
        conn.commit()
        conn.close()
    
    def get_prediction_stats(self):
        """Get prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT 
                predicted_category,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                DATE(timestamp) as date
            FROM predictions
            GROUP BY predicted_category, DATE(timestamp)
            ORDER BY timestamp DESC
        """, conn)
        conn.close()
        return df
    
    def save_model_metadata(self, metadata):
        """Save model metadata and return model_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate other models
        cursor.execute("UPDATE model_metadata SET is_active = 0")
        
        # Insert new model
        cursor.execute("""
            INSERT INTO model_metadata 
            (model_name, accuracy, f1_score, precision_score, recall_score, 
             trained_date, training_time, n_samples, n_train, n_test, framework, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            metadata['model_type'],
            metadata['accuracy'],
            metadata['f1_score'],
            metadata['precision'],
            metadata['recall'],
            metadata['trained_date'],
            metadata['training_time'],
            metadata['n_samples'],
            metadata['n_train'],
            metadata['n_test'],
            metadata['framework']
        ))
        
        model_id = cursor.lastrowid
        
        # Save category metrics
        if 'category_metrics' in metadata:
            for cat, metrics in metadata['category_metrics'].items():
                cursor.execute("""
                    INSERT INTO category_metrics 
                    (model_id, category, precision_val, recall_val, f1_score, support)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    cat,
                    metrics.get('precision', 0.0),
                    metrics.get('recall', 0.0),
                    metrics.get('f1-score', 0.0),
                    metrics.get('support', 0)
                ))
        
        conn.commit()
        conn.close()
        return model_id
    
    def get_model_history(self):
        """Get all model training history"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT 
                model_name,
                accuracy,
                f1_score,
                trained_date,
                n_samples,
                training_time,
                is_active
            FROM model_metadata
            ORDER BY trained_date DESC
        """, conn)
        conn.close()
        return df
    
    def get_category_metrics(self, model_id):
        """Get category metrics for a specific model"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT 
                category,
                precision_val,
                recall_val,
                f1_score,
                support
            FROM category_metrics
            WHERE model_id = ?
            ORDER BY category
        """, conn, params=(model_id,))
        conn.close()
        return df


class PESTLESparkModel:
    """Production PESTLE classifier with Apache Spark"""
    
    def __init__(self):
        self.spark = None
        self.pipeline_model = None
        self.label_indexer = None
        self.categories = ['Political', 'Economic', 'Social', 'Technological', 'Legal', 'Environmental']
        self.metadata = {}
        self.model_dir = Path("pestle_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.pestle_keywords = {
            'Political': ['government', 'election', 'policy', 'congress', 'senate', 
                         'president', 'legislation', 'vote', 'parliament', 'diplomacy',
                         'democratic', 'republican', 'political', 'minister', 'governor'],
            'Economic': ['economy', 'market', 'stock', 'trade', 'gdp', 'inflation',
                        'interest rate', 'unemployment', 'fed', 'revenue', 'profit',
                        'bank', 'financial', 'investment', 'dollar', 'economic'],
            'Social': ['healthcare', 'education', 'social', 'community', 'demographic',
                      'population', 'immigration', 'diversity', 'equality', 'housing',
                      'welfare', 'pension', 'family', 'culture', 'society'],
            'Technological': ['technology', 'ai', 'artificial intelligence', 'innovation',
                            'digital', 'cyber', 'data', 'software', 'internet', 'automation',
                            'robot', 'computing', 'tech', 'app', 'platform'],
            'Legal': ['law', 'court', 'legal', 'lawsuit', 'judge', 'attorney',
                     'regulation', 'compliance', 'contract', 'patent', 'trial',
                     'justice', 'ruling', 'verdict', 'litigation', 'statute'],
            'Environmental': ['climate', 'environment', 'carbon', 'emission', 'pollution',
                            'renewable', 'energy', 'sustainability', 'green', 'conservation',
                            'solar', 'wind', 'ecological', 'wildlife', 'ecosystem']
        }
        
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = SparkSession.builder \
                .appName("PESTLE_Classifier") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.sql.shuffle.partitions", "8") \
                .config("spark.driver.maxResultSize", "2g") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("ERROR")
            return True
        except Exception as e:
            st.error(f"❌ Could not initialize Spark: {e}")
            return False
    
    def train(self, df, progress_callback=None):
        """Train model with Apache Spark"""
        start_time = datetime.now()
        
        if progress_callback:
            progress_callback("Converting to Spark DataFrame...")
        
        # Convert to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        if progress_callback:
            progress_callback("Preprocessing text data...")
        
        # Prepare combined text column
        spark_df = spark_df.withColumn(
            'text',
            F.lower(
                F.concat_ws(' ', 
                    F.coalesce(F.col('Headline'), F.lit('')),
                    F.coalesce(F.col('Description'), F.lit('')),
                    F.coalesce(F.col('Topic_Tags'), F.lit(''))
                )
            )
        )
        
        # Remove nulls and empty strings
        spark_df = spark_df.filter(F.col('text').isNotNull())
        spark_df = spark_df.filter(F.length(F.col('text')) > 0)
        
        if progress_callback:
            progress_callback("Encoding labels...")
        
        # String indexer for labels
        self.label_indexer = StringIndexer(inputCol="PESTLE_Category", outputCol="label")
        indexer_model = self.label_indexer.fit(spark_df)
        spark_df = indexer_model.transform(spark_df)
        
        if progress_callback:
            progress_callback("Splitting dataset (80-20)...")
        
        # Split data (80-20)
        train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
        
        # Cache for performance
        train_df.cache()
        test_df.cache()
        
        n_train = train_df.count()
        n_test = test_df.count()
        
        if progress_callback:
            progress_callback(f"Building ML pipeline (Train: {n_train}, Test: {n_test})...")
        
        # Build ML pipeline
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
        idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)
        
        # Random Forest with optimized parameters
        rf = RandomForestClassifier(
            labelCol="label", 
            featuresCol="features", 
            numTrees=200,
            maxDepth=20,
            maxBins=32,
            minInstancesPerNode=1,
            seed=42
        )
        
        # Label converter (index back to string)
        labelConverter = IndexToString(
            inputCol="prediction",
            outputCol="predictedCategory",
            labels=indexer_model.labels
        )
        
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, rf, labelConverter])
        
        if progress_callback:
            progress_callback("Training Random Forest model (this may take a few minutes)...")
        
        # Train model
        self.pipeline_model = pipeline.fit(train_df)
        
        if progress_callback:
            progress_callback("Evaluating model performance...")
        
        # Evaluate on test set
        predictions = self.pipeline_model.transform(test_df)
        
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction"
        )
        
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        
        if progress_callback:
            progress_callback("Calculating per-category metrics...")
        
        # Get per-category metrics
        pred_pandas = predictions.select("label", "prediction", "PESTLE_Category").toPandas()
        category_metrics = self._calculate_category_metrics(pred_pandas)
        
        # Get confusion matrix data
        confusion_data = self._get_confusion_matrix(pred_pandas)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store metadata
        self.metadata = {
            'model_type': 'Spark Random Forest',
            'accuracy': float(accuracy),
            'f1_score': float(f1_score),
            'precision': float(precision),
            'recall': float(recall),
            'trained_date': datetime.now().isoformat(),
            'training_time': training_time,
            'n_samples': df.shape[0],
            'n_train': int(n_train),
            'n_test': int(n_test),
            'categories': self.categories,
            'framework': 'Apache Spark',
            'category_metrics': category_metrics,
            'confusion_matrix': confusion_data,
            'label_mapping': {label: idx for idx, label in enumerate(indexer_model.labels)}
        }
        
        # Clean up cache
        train_df.unpersist()
        test_df.unpersist()
        
        if progress_callback:
            progress_callback("Training complete! ✅")
        
        return self.metadata
    
    def _calculate_category_metrics(self, pred_df):
        """Calculate per-category metrics"""
        from sklearn.metrics import classification_report
        
        report = classification_report(
            pred_df['PESTLE_Category'], 
            pred_df.apply(lambda row: self._get_category_from_label(row['prediction']), axis=1),
            output_dict=True,
            zero_division=0
        )
        
        category_metrics = {}
        for cat in self.categories:
            if cat in report:
                category_metrics[cat] = {
                    'precision': report[cat]['precision'],
                    'recall': report[cat]['recall'],
                    'f1-score': report[cat]['f1-score'],
                    'support': report[cat]['support']
                }
        
        return category_metrics
    
    def _get_category_from_label(self, label_idx):
        """Convert label index to category name"""
        label_map = self.metadata.get('label_mapping', {})
        for cat, idx in label_map.items():
            if idx == label_idx:
                return cat
        return "Unknown"
    
    def _get_confusion_matrix(self, pred_df):
        """Generate confusion matrix data"""
        from sklearn.metrics import confusion_matrix
        
        y_true = pred_df['PESTLE_Category']
        y_pred = pred_df.apply(lambda row: self._get_category_from_label(row['prediction']), axis=1)
        
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        return {
            'matrix': cm.tolist(),
            'labels': self.categories
        }
    
    def save(self, model_name="pestle_spark_model"):
        """Save model to disk"""
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        # Save Spark pipeline
        spark_model_path = str(model_path / "spark_pipeline")
        self.pipeline_model.write().overwrite().save(spark_model_path)
        
        # Save metadata
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return str(model_path)
    
    def load(self, model_name="pestle_spark_model"):
        """Load model from disk"""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load Spark pipeline
        spark_model_path = str(model_path / "spark_pipeline")
        self.pipeline_model = PipelineModel.load(spark_model_path)
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        return True
    
    def predict(self, text):
        """Predict PESTLE category"""
        if self.pipeline_model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Create Spark DataFrame
        test_df = self.spark.createDataFrame([(text,)], ["text"])
        
        # Make prediction
        prediction = self.pipeline_model.transform(test_df)
        pred_row = prediction.select("prediction", "probability", "predictedCategory").collect()[0]
        
        predicted_category = pred_row["predictedCategory"]
        
        # Extract probabilities
        probabilities = pred_row["probability"].toArray()
        
        # Map probabilities to categories
        label_map = self.metadata.get('label_mapping', {})
        prob_dict = {}
        for cat, idx in label_map.items():
            if idx < len(probabilities):
                prob_dict[cat] = float(probabilities[int(idx)])
        
        return {
            'category': predicted_category,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict
        }
    
    def predict_batch(self, texts):
        """Predict multiple texts at once"""
        if self.pipeline_model is None:
            raise ValueError("Model not loaded.")
        
        # Create Spark DataFrame
        test_df = self.spark.createDataFrame([(t,) for t in texts], ["text"])
        
        # Make predictions
        predictions = self.pipeline_model.transform(test_df)
        results = predictions.select("text", "predictedCategory", "probability").collect()
        
        batch_results = []
        for row in results:
            probabilities = row["probability"].toArray()
            batch_results.append({
                'text': row["text"],
                'category': row["predictedCategory"],
                'confidence': float(max(probabilities))
            })
        
        return batch_results
    
    def stop_spark(self):
        """
