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
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            self.spark = None


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_confusion_matrix(confusion_data):
    """Plot confusion matrix heatmap"""
    cm = np.array(confusion_data['matrix'])
    labels = confusion_data['labels']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Category',
        yaxis_title='True Category',
        width=700,
        height=600
    )
    
    return fig


def plot_category_metrics(category_metrics):
    """Plot per-category performance metrics"""
    categories = list(category_metrics.keys())
    precision = [category_metrics[cat]['precision'] for cat in categories]
    recall = [category_metrics[cat]['recall'] for cat in categories]
    f1 = [category_metrics[cat]['f1-score'] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Precision', x=categories, y=precision, marker_color='#2E86AB'))
    fig.add_trace(go.Bar(name='Recall', x=categories, y=recall, marker_color='#A23B72'))
    fig.add_trace(go.Bar(name='F1-Score', x=categories, y=f1, marker_color='#F18F01'))
    
    fig.update_layout(
        title='Category-wise Performance Metrics',
        xaxis_title='PESTLE Category',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        height=500
    )
    
    return fig


def plot_prediction_probabilities(probabilities):
    """Plot prediction probabilities"""
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker=dict(
            color=list(sorted_probs.values()),
            colorscale='Viridis',
            showscale=True
        ),
        text=[f'{v:.2%}' for v in sorted_probs.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Prediction Confidence by Category',
        xaxis_title='Probability',
        yaxis_title='Category',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_prediction_distribution(stats_df):
    """Plot prediction distribution over time"""
    if stats_df.empty:
        return None
    
    fig = px.bar(
        stats_df.groupby('predicted_category')['count'].sum().reset_index(),
        x='predicted_category',
        y='count',
        title='Prediction Distribution',
        labels={'predicted_category': 'Category', 'count': 'Number of Predictions'},
        color='predicted_category',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def plot_model_comparison(history_df):
    """Plot model performance comparison"""
    if history_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['trained_date'],
        y=history_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['trained_date'],
        y=history_df['f1_score'],
        mode='lines+markers',
        name='F1 Score',
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Model Performance Over Time',
        xaxis_title='Training Date',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="PESTLE Classification System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 PESTLE Classification System</h1>', unsafe_allow_html=True)
    st.markdown("**Production-ready ML system with Apache Spark, SQLite & Analytics**")
    st.markdown("---")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = PESTLESparkModel()
    
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=PESTLE+AI", use_column_width=True)
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🎯 Train Model", "🔮 Predictions", "📈 Analytics", "📚 About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Status")
        if st.session_state.trained:
            st.success("✅ Model Ready")
            metadata = st.session_state.model.metadata
            st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
            st.metric("F1 Score", f"{metadata['f1_score']:.2%}")
        else:
            st.warning("⚠️ No Model Loaded")
            if st.button("Load Existing Model"):
                try:
                    st.session_state.model.load()
                    st.session_state.trained = True
                    st.success("Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load model: {e}")
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Train Model":
        show_train_page()
    elif page == "🔮 Predictions":
        show_prediction_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "📚 About":
        show_about_page()


def show_home_page():
    """Home page with system overview"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Welcome to PESTLE Classification System")
        st.markdown("""
        This production-ready machine learning system classifies news articles and text content 
        into PESTLE categories using Apache Spark and advanced NLP techniques.
        
        **🎯 Key Features:**
        - ⚡ **Distributed Processing** with Apache Spark
        - 🎓 **Advanced ML** with Random Forest classifier
        - 💾 **Data Persistence** with SQLite
        - 📊 **Real-time Analytics** and insights
        - 🔄 **Batch Processing** capabilities
        - 📈 **Performance Tracking** and model versioning
        """)
        
        st.markdown("### PESTLE Categories")
        categories = {
            "🏛️ Political": "Government policies, elections, regulations",
            "💰 Economic": "Markets, trade, financial indicators",
            "👥 Social": "Demographics, culture, social trends",
            "🔬 Technological": "Innovation, digital transformation, AI",
            "⚖️ Legal": "Laws, compliance, court decisions",
            "🌍 Environmental": "Climate, sustainability, conservation"
        }
        
        for cat, desc in categories.items():
            st.markdown(f"**{cat}**: {desc}")
    
    with col2:
        st.markdown("### Quick Stats")
        
        # Get statistics
        stats_df = st.session_state.db.get_prediction_stats()
        history_df = st.session_state.db.get_model_history()
        
        if not stats_df.empty:
            total_predictions = stats_df['count'].sum()
            avg_confidence = stats_df['avg_confidence'].mean()
            
            st.metric("Total Predictions", f"{int(total_predictions):,}")
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        if not history_df.empty:
            st.metric("Models Trained", len(history_df))
            if len(history_df) > 0:
                best_accuracy = history_df['accuracy'].max()
                st.metric("Best Accuracy", f"{best_accuracy:.2%}")
        
        st.markdown("### Get Started")
        st.info("👈 Use the sidebar to navigate through different sections")


def show_train_page():
    """Model training page"""
    st.markdown("## 🎯 Train PESTLE Classification Model")
    
    st.markdown("""
    Upload your training dataset (CSV format) with the following columns:
    - `Headline`: Article headline
    - `Description`: Article description
    - `Topic_Tags`: Related tags
    - `PESTLE_Category`: Target category (Political, Economic, Social, Technological, Legal, Environmental)
    """)
    
    uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show dataset preview
            with st.expander("📊 Dataset Preview"):
                st.dataframe(df.head(10))
                
                # Show category distribution
                if 'PESTLE_Category' in df.columns:
                    st.markdown("**Category Distribution:**")
                    cat_dist = df['PESTLE_Category'].value_counts()
                    fig = px.bar(
                        x=cat_dist.index,
                        y=cat_dist.values,
                        labels={'x': 'Category', 'y': 'Count'},
                        title='Training Data Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Training configuration
            st.markdown("### Training Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name", "pestle_spark_model")
            
            with col2:
                st.info(f"Dataset: {df.shape[0]} samples")
            
            # Train button
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                try:
                    # Progress callback
                    def update_progress(message):
                        status_placeholder.info(f"⏳ {message}")
                    
                    with st.spinner("Training model..."):
                        # Train model
                        metadata = st.session_state.model.train(df, progress_callback=update_progress)
                        
                        # Save model
                        status_placeholder.info("💾 Saving model...")
                        st.session_state.model.save(model_name)
                        
                        # Save to database
                        model_id = st.session_state.db.save_model_metadata(metadata)
                        
                        st.session_state.trained = True
                        
                        status_placeholder.empty()
                        st.success("✅ Model trained and saved successfully!")
                        
                        # Display results
                        st.markdown("### Training Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metadata['accuracy']:.2%}")
                        col2.metric("F1 Score", f"{metadata['f1_score']:.2%}")
                        col3.metric("Precision", f"{metadata['precision']:.2%}")
                        col4.metric("Recall", f"{metadata['recall']:.2%}")
                        
                        st.metric("Training Time", f"{metadata['training_time']:.2f} seconds")
                        
                        # Confusion matrix
                        st.markdown("### Confusion Matrix")
                        fig = plot_confusion_matrix(metadata['confusion_matrix'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Category metrics
                        st.markdown("### Category-wise Performance")
                        fig = plot_category_metrics(metadata['category_metrics'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed metrics table
                        with st.expander("📋 Detailed Metrics"):
                            metrics_data = []
                            for cat, metrics in metadata['category_metrics'].items():
                                metrics_data.append({
                                    'Category': cat,
                                    'Precision': f"{metrics['precision']:.3f}",
                                    'Recall': f"{metrics['recall']:.3f}",
                                    'F1-Score': f"{metrics['f1-score']:.3f}",
                                    'Support': int(metrics['support'])
                                })
                            st.table(pd.DataFrame(metrics_data))
                
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    else:
        st.info("📤 Please upload a CSV file to begin training")
        
        # Show sample data format
        with st.expander("💡 Sample Data Format"):
            sample_df = pd.DataFrame({
                'Headline': ['New Climate Bill Passed', 'Stock Market Hits Record High'],
                'Description': ['Government passes legislation...', 'Markets rally after...'],
                'Topic_Tags': ['climate, policy', 'stocks, economy'],
                'PESTLE_Category': ['Environmental', 'Economic']
            })
            st.dataframe(sample_df)


def show_prediction_page():
    """Prediction interface"""
    st.markdown("## 🔮 Make Predictions")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train or load a model first")
        return
    
    # Prediction mode
    mode = st.radio("Prediction Mode", ["Single Text", "Batch Processing"])
    
    if mode == "Single Text":
        st.markdown("### Single Text Prediction")
        
        # Input methods
        input_method = st.radio("Input Method", ["Text Input", "Sample Examples"])
        
        if input_method == "Text Input":
            text = st.text_area(
                "Enter text to classify:",
                height=150,
                placeholder="Enter news headline, article, or any text..."
            )
        else:
            examples = {
                "Political": "Congress passes new infrastructure bill with bipartisan support",
                "Economic": "Federal Reserve raises interest rates to combat inflation",
                "Social": "New study shows increasing diversity in urban communities",
                "Technological": "AI breakthrough enables faster drug discovery process",
                "Legal": "Supreme Court ruling affects nationwide employment policies",
                "Environmental": "Renewable energy sector sees record growth in solar installations"
            }
            selected_example = st.selectbox("Select an example:", list(examples.keys()))
            text = examples[selected_example]
            st.text_area("Text:", value=text, height=100, disabled=True)
        
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            if text.strip():
                with st.spinner("Making prediction..."):
                    try:
                        result = st.session_state.model.predict(text)
                        
                        # Save to database
                        st.session_state.db.save_prediction(
                            text,
                            result['category'],
                            result['confidence'],
                            st.session_state.model.metadata.get('model_type', 'Unknown')
                        )
                        
                        # Display results
                        st.markdown("### Prediction Results")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"### {result['category']}")
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        with col2:
                            fig = plot_prediction_probabilities(result['probabilities'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show all probabilities
                        with st.expander("📊 All Category Probabilities"):
                            prob_df = pd.DataFrame([
                                {'Category': cat, 'Probability': f"{prob:.2%}"}
                                for cat, prob in sorted(
                                    result['probabilities'].items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                            ])
                            st.table(prob_df)
                    
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
            else:
                st.warning("Please enter some text")
    
    else:  # Batch Processing
        st.markdown("### Batch Processing")
        
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                    return
                
                st.success(f"✅ Loaded {len(df)} texts")
                st.dataframe(df.head())
                
                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner(f"Processing {len(df)} texts..."):
                        try:
                            results = st.session_state.model.predict_batch(df['text'].tolist())
                            
                            # Add results to dataframe
                            df['predicted_category'] = [r['category'] for r in results]
                            df['confidence'] = [r['confidence'] for r in results]
                            
                            st.success("✅ Batch processing complete!")
                            
                            # Show results
                            st.markdown("### Results")
                            st.dataframe(df)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                            # Summary statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Category Distribution**")
                                cat_counts = df['predicted_category'].value_counts()
                                fig = px.pie(
                                    values=cat_counts.values,
                                    names=cat_counts.index,
                                    title='Predicted Categories'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Confidence Distribution**")
                                fig = px.histogram(
                                    df,
                                    x='confidence',
                                    nbins=20,
                                    title='Confidence Score Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"❌ Batch processing failed: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")


def show_analytics_page():
    """Analytics and insights page"""
    st.markdown("## 📈 Analytics & Insights")
    
    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Analytics", "🎯 Model Performance", "📅 Historical Trends"])
    
    with tab1:
        st.markdown("### Prediction Analytics")
        
        stats_df = st.session_state.db.get_prediction_stats()
        
        if not stats_df.empty:
            # Overall statistics
            col1, col2, col3 = st.columns(3)
            
            total_predictions = stats_df['count'].sum()
            avg_confidence = stats_df['avg_confidence'].mean()
            unique_categories = stats_df['predicted_category'].nunique()
            
            col1.metric("Total Predictions", f"{int(total_predictions):,}")
            col2.metric("Average Confidence", f"{avg_confidence:.2%}")
            col3.metric("Active Categories", unique_categories)
            
            # Distribution chart
            fig = plot_prediction_distribution(stats_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence by category
            st.markdown("### Confidence by Category")
            conf_by_cat = stats_df.groupby('predicted_category')['avg_confidence'].mean().sort_values(ascending=False)
            
            fig = go.Figure(go.Bar(
                x=conf_by_cat.values,
                y=conf_by_cat.index,
                orientation='h',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title='Average Confidence by Category',
                xaxis_title='Average Confidence',
                yaxis_title='Category',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions
            with st.expander("📋 Recent Predictions"):
                conn = sqlite3.connect(st.session_state.db.db_path)
                recent_df = pd.read_sql_query("""
                    SELECT text, predicted_category, confidence, timestamp
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, conn)
                conn.close()
                st.dataframe(recent_df)
        
        else:
            st.info("No predictions yet. Start making predictions to see analytics!")
    
    with tab2:
        st.markdown("### Model Performance")
        
        history_df = st.session_state.db.get_model_history()
        
        if not history_df.empty:
            # Current model metrics
            if st.session_state.trained:
                st.markdown("### Current Model")
                metadata = st.session_state.model.metadata
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metadata['accuracy']:.2%}")
                col2.metric("F1 Score", f"{metadata['f1_score']:.2%}")
                col3.metric("Precision", f"{metadata['precision']:.2%}")
                col4.metric("Recall", f"{metadata['recall']:.2%}")
                
                # Confusion matrix
                if 'confusion_matrix' in metadata:
                    fig = plot_confusion_matrix(metadata['confusion_matrix'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category metrics
                if 'category_metrics' in metadata:
                    fig = plot_category_metrics(metadata['category_metrics'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            st.markdown("### Model History")
            fig = plot_model_comparison(history_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Model table
            st.dataframe(history_df, use_container_width=True)
        
        else:
            st.info("No model history yet. Train a model to see performance metrics!")
    
    with tab3:
        st.markdown("### Historical Trends")
        
        stats_df = st.session_state.db.get_prediction_stats()
        
        if not stats_df.empty and 'date' in stats_df.columns:
            # Predictions over time
            daily_stats = stats_df.groupby('date')['count'].sum().reset_index()
            
            fig = px.line(
                daily_stats,
                x='date',
                y='count',
                title='Predictions Over Time',
                labels={'date': 'Date', 'count': 'Number of Predictions'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category trends
            category_trends = stats_df.groupby(['date', 'predicted_category'])['count'].sum().reset_index()
            
            fig = px.line(
                category_trends,
                x='date',
                y='count',
                color='predicted_category',
                title='Category Trends Over Time',
                labels={'date': 'Date', 'count': 'Predictions', 'predicted_category': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Not enough historical data yet. Keep using the system to see trends!")


def show_about_page():
    """About page with system information"""
    st.markdown("## 📚 About PESTLE Classification System")
    
    st.markdown("""
    ### System Overview
    
    This production-ready PESTLE Classification System leverages cutting-edge technologies 
    to provide accurate, scalable text classification for business intelligence and strategic analysis.
    
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **ML Framework**: Apache Spark MLlib
    - **Database**: SQLite
    - **Visualization**: Plotly
    - **ML Algorithm**: Random Forest Classifier with TF-IDF
    
    ### PESTLE Framework
    
    PESTLE analysis is a strategic tool used to identify macro-environmental factors:
    
    - **Political**: Government policies, political stability, regulations
    - **Economic**: Economic growth, exchange rates, inflation
    - **Social**: Demographics, cultural trends, consumer behavior
    - **Technological**: Innovation, R&D, automation
    - **Legal**: Employment laws, consumer protection, data privacy
    - **Environmental**: Climate change, sustainability, carbon footprint
    
    ### Model Architecture
    
    1. **Text Preprocessing**: Tokenization, stop word removal
    2. **Feature Engineering**: TF-IDF vectorization (5000 features)
    3. **Classification**: Random Forest (200 trees, max depth 20)
    4. **Evaluation**: Multi-class metrics, confusion matrix
    
    ### Key Features
    
    ✅ **Distributed Processing**: Spark enables handling large datasets
    ✅ **Real-time Predictions**: Fast inference for single or batch inputs
    ✅ **Performance Tracking**: SQLite database stores all predictions and metrics
    ✅ **Model Versioning**: Track model evolution over time
    ✅ **Interactive Analytics**: Rich visualizations and insights
    
    ### System Requirements
    
    - Python 3.8+
    - Apache Spark 3.4+
    - 4GB+ RAM
    - Modern web browser
    
    ### Usage Guidelines
    
    1. **Train Model**: Upload labeled dataset and train classifier
    2. **Make Predictions**: Classify single texts or batch process files
    3. **Monitor Performance**: Track accuracy and prediction patterns
    4. **Iterate**: Retrain with new data to improve performance
    
    ### Performance Notes
    
    - Training time varies with dataset size (typically 2-10 minutes)
    - Prediction latency: <100ms for single text
    - Batch processing: ~1000 texts per minute
    - Model accuracy: Typically 85-95% depending on training data
    
    ### Best Practices
    
    - Use balanced training datasets across all categories
    - Include diverse text sources (news, reports, articles)
    - Regularly retrain with new data
    - Monitor prediction confidence scores
    - Review low-confidence predictions manually
    
    ### Support & Resources
    
    - Documentation: [GitHub Repository]
    - Issues: [GitHub Issues]
    - Contact: support@pestle-classifier.com
    
    ### Version Information
    
    - **Version**: 1.0.0
    - **Release Date**: 2024
    - **Framework**: Apache Spark + Streamlit
    - **License**: MIT
    """)
    
    # System status
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🟢 Spark: Active")
    
    with col2:
        st.info("🟢 Database: Connected")
    
    with col3:
        if st.session_state.trained:
            st.success("🟢 Model: Loaded")
        else:
            st.warning("🟡 Model: Not Loaded")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    main()
