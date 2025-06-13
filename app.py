# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
from threading import Thread
import time
import schedule
import mysql.connector
from mysql.connector import Error
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'iwfy4jjr74hioovop%&%&yg67Tudbc8h2'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'etl_pipeline',
    'user': 'root',
    'password': 'your_mysql_password',  # Change this
}

# Email configuration for notifications
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your_email@gmail.com',  # Change this
    'sender_password': 'your_app_password',  # Change this
    'recipient_email': 'recipient@gmail.com'  # Change this
}

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

class DataCleaner:
    """Comprehensive data cleaning class"""
    
    def __init__(self):
        self.cleaning_report = {
            'original_rows': 0,
            'final_rows': 0,
            'columns_processed': 0,
            'duplicates_removed': 0,
            'missing_values_handled': 0,
            'outliers_removed': 0,
            'data_types_converted': 0,
            'cleaning_steps': []
        }
    
    def clean_dataset(self, df):
        """Main cleaning function"""
        logger.info("Starting data cleaning process")
        self.cleaning_report['original_rows'] = len(df)
        self.cleaning_report['columns_processed'] = len(df.columns)
        
        # Step 1: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 2: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 3: Handle outliers
        df = self._handle_outliers(df)
        
        # Step 4: Standardize data types
        df = self._standardize_data_types(df)
        
        # Step 5: Clean text data
        df = self._clean_text_data(df)
        
        # Step 6: Validate and fix dates
        df = self._clean_date_columns(df)
        
        self.cleaning_report['final_rows'] = len(df)
        logger.info(f"Data cleaning completed. Processed {self.cleaning_report['original_rows']} -> {self.cleaning_report['final_rows']} rows")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values based on column type"""
        missing_before = df.isnull().sum().sum()
        
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[column].fillna(df[column].median(), inplace=True)
                elif df[column].dtype == 'object':
                    # Fill categorical columns with mode
                    mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                    df[column].fillna(mode_value, inplace=True)
                elif df[column].dtype == 'datetime64[ns]':
                    # Forward fill for datetime columns
                    df[column].fillna(method='ffill', inplace=True)
        
        missing_after = df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        self.cleaning_report['cleaning_steps'].append(f"Handled {missing_before - missing_after} missing values")
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        duplicates_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - len(df)
        
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        self.cleaning_report['cleaning_steps'].append(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def _handle_outliers(self, df):
        """Remove outliers using IQR method for numeric columns"""
        outliers_removed = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(df)
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            outliers_removed += outliers_before - len(df)
        
        self.cleaning_report['outliers_removed'] = outliers_removed
        self.cleaning_report['cleaning_steps'].append(f"Removed {outliers_removed} outlier records")
        
        return df
    
    def _standardize_data_types(self, df):
        """Automatically detect and convert data types"""
        conversions = 0
        
        for column in df.columns:
            # Try to convert to numeric if possible
            if df[column].dtype == 'object':
                # Check if it's a numeric string
                try:
                    pd.to_numeric(df[column], errors='raise')
                    df[column] = pd.to_numeric(df[column])
                    conversions += 1
                except:
                    # Check if it's a date string
                    try:
                        pd.to_datetime(df[column], errors='raise')
                        df[column] = pd.to_datetime(df[column])
                        conversions += 1
                    except:
                        pass
        
        self.cleaning_report['data_types_converted'] = conversions
        self.cleaning_report['cleaning_steps'].append(f"Converted {conversions} columns to appropriate data types")
        
        return df
    
    def _clean_text_data(self, df):
        """Clean text columns"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            # Remove leading/trailing whitespace
            df[column] = df[column].astype(str).str.strip()
            # Standardize case (title case for names, etc.)
            if 'name' in column.lower():
                df[column] = df[column].str.title()
            # Remove special characters if needed
            df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True)
        
        self.cleaning_report['cleaning_steps'].append("Cleaned text data (whitespace, case, special characters)")
        
        return df
    
    def _clean_date_columns(self, df):
        """Clean and validate date columns"""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        for column in date_columns:
            # Remove future dates if they seem unrealistic
            future_cutoff = datetime.now() + timedelta(days=365)
            df = df[df[column] <= future_cutoff]
            
            # Remove very old dates if they seem unrealistic
            past_cutoff = datetime(1900, 1, 1)
            df = df[df[column] >= past_cutoff]
        
        if len(date_columns) > 0:
            self.cleaning_report['cleaning_steps'].append(f"Validated {len(date_columns)} date columns")
        
        return df

# DatabaseManager for SQLite (No MySQL Required)
class DatabaseManager:
    """SQLite database operations"""

    def __init__(self):
        self.engine = None
        self.connection = None

    def connect(self):
        """Connect to SQLite database"""
        try:
            self.engine = create_engine('sqlite:///etl_pipeline.db')
            self.connection = sqlite3.connect('etl_pipeline.db')
            logger.info("Successfully connected to SQLite database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to SQLite: {e}")
            return False

    def create_tables(self):
        """Create necessary tables in SQLite"""
        if not self.connection:
            return False
        try:
            cursor = self.connection.cursor()

            # Create processing_jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    original_rows INTEGER,
                    processed_rows INTEGER,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    cleaning_report TEXT
                )
            """)

            # Create scheduled_jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT,
                    file_path TEXT,
                    schedule_pattern TEXT,
                    last_run TIMESTAMP,
                    next_run TIMESTAMP,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.connection.commit()
            logger.info("Database tables created successfully in SQLite")
            return True

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

    def save_processing_job(self, job_data):
        """Save processing job details"""
        if not self.connection:
            return False
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO processing_jobs 
                (filename, original_rows, processed_rows, processing_time, status, cleaning_report)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            values = (
                job_data['filename'],
                job_data['original_rows'],
                job_data['processed_rows'],
                job_data['processing_time'],
                job_data['status'],
                json.dumps(job_data['cleaning_report'])
            )

            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Processing job saved: {job_data['filename']}")
            return True

        except Exception as e:
            logger.error(f"Error saving processing job: {e}")
            return False

    def save_data_to_table(self, df, table_name):
        """Save cleaned data to SQLite table"""
        if not self.engine:
            return False

        try:
            table_name = table_name.replace('.', '_').replace('-', '_').lower()
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Data saved to SQLite table: {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving data to SQLite table: {e}")
            return False

class DataVisualizer:
    """Create various data visualizations"""
    
    def __init__(self):
        self.plots = {}
    
    def generate_comprehensive_report(self, df, filename):
        """Generate comprehensive data visualization report"""
        self.plots = {}
        
        # Basic statistics
        self._create_summary_stats(df)
        
        # Distribution plots
        self._create_distribution_plots(df)
        
        # Correlation heatmap
        self._create_correlation_heatmap(df)
        
        # Missing values visualization
        self._create_missing_values_plot(df)
        
        # Time series plots if date columns exist
        self._create_time_series_plots(df)
        
        # Save plots
        self._save_plots(filename)
        
        return self.plots
    
    def _create_summary_stats(self, df):
        """Create summary statistics table"""
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary = numeric_df.describe()
            
            fig = go.Figure(data=[go.Table(
                header=dict(values=['Statistic'] + list(summary.columns)),
                cells=dict(values=[summary.index] + [summary[col] for col in summary.columns])
            )])
            
            self.plots['summary_stats'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_distribution_plots(self, df):
        """Create distribution plots for numeric columns"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to first 4
        
        if len(numeric_columns) > 0:
            fig = go.Figure()
            
            for col in numeric_columns:
                fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.7))
            
            fig.update_layout(
                title="Distribution of Numeric Variables",
                xaxis_title="Value",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            
            self.plots['distributions'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_correlation_heatmap(self, df):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(title="Correlation Heatmap")
            self.plots['correlation'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_missing_values_plot(self, df):
        """Create missing values visualization"""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = go.Figure([go.Bar(x=missing_data.index, y=missing_data.values)])
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Columns",
                yaxis_title="Number of Missing Values"
            )
            
            self.plots['missing_values'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_time_series_plots(self, df):
        """Create time series plots if date columns exist"""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(date_columns) > 0 and len(numeric_columns) > 0:
            date_col = date_columns[0]
            numeric_col = numeric_columns[0]
            
            # Sort by date
            df_sorted = df.sort_values(date_col)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_sorted[date_col],
                y=df_sorted[numeric_col],
                mode='lines+markers',
                name=f'{numeric_col} over time'
            ))
            
            fig.update_layout(
                title=f"Time Series: {numeric_col}",
                xaxis_title="Date",
                yaxis_title=numeric_col
            )
            
            self.plots['time_series'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _save_plots(self, filename):
        """Save plots as static images"""
        try:
            # This would save static versions of plots
            # Implementation depends on your needs
            pass
        except Exception as e:
            logger.error(f"Error saving plots: {e}")

class EmailNotifier:
    """Send email notifications"""
    
    def send_processing_complete_email(self, job_data):
        """Send email when processing is complete"""
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = EMAIL_CONFIG['recipient_email']
            msg['Subject'] = f"ETL Processing Complete - {job_data['filename']}"
            
            body = f"""
            ETL Processing Completed Successfully!
            
            File: {job_data['filename']}
            Original Rows: {job_data['original_rows']}
            Processed Rows: {job_data['processed_rows']}
            Processing Time: {job_data['processing_time']:.2f} seconds
            
            Cleaning Summary:
            - Duplicates Removed: {job_data['cleaning_report'].get('duplicates_removed', 0)}
            - Missing Values Handled: {job_data['cleaning_report'].get('missing_values_handled', 0)}
            - Outliers Removed: {job_data['cleaning_report'].get('outliers_removed', 0)}
            
            Best regards,
            ETL Pipeline System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            text = msg.as_string()
            server.sendmail(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['recipient_email'], text)
            server.quit()
            
            logger.info("Processing complete email sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")

class ScheduleManager:
    """Manage scheduled ETL jobs"""
    
    def __init__(self):
        self.jobs = {}
    
    def schedule_job(self, job_name, file_path, schedule_pattern):
        """Schedule a new ETL job"""
        try:
            # Parse schedule pattern (e.g., "daily", "weekly", "0 9 * * *")
            if schedule_pattern == "daily":
                scheduler.add_job(
                    func=self.run_scheduled_etl,
                    trigger="cron",
                    hour=9,
                    minute=0,
                    args=[file_path],
                    id=job_name
                )
            elif schedule_pattern == "weekly":
                scheduler.add_job(
                    func=self.run_scheduled_etl,
                    trigger="cron",
                    day_of_week=1,
                    hour=9,
                    minute=0,
                    args=[file_path],
                    id=job_name
                )
            else:
                # Custom cron pattern
                parts = schedule_pattern.split()
                if len(parts) == 5:
                    scheduler.add_job(
                        func=self.run_scheduled_etl,
                        trigger="cron",
                        minute=parts[0],
                        hour=parts[1],
                        day=parts[2],
                        month=parts[3],
                        day_of_week=parts[4],
                        args=[file_path],
                        id=job_name
                    )
            
            self.jobs[job_name] = {
                'file_path': file_path,
                'schedule_pattern': schedule_pattern,
                'status': 'active'
            }
            
            logger.info(f"Scheduled job '{job_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling job: {e}")
            return False
    
    def run_scheduled_etl(self, file_path):
        """Run ETL process for scheduled job"""
        try:
            logger.info(f"Running scheduled ETL for: {file_path}")
            
            # Read file
            df = pd.read_csv(file_path)
            
            # Clean data
            cleaner = DataCleaner()
            cleaned_df = cleaner.clean_dataset(df)
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"scheduled_{timestamp}_{os.path.basename(file_path)}"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            cleaned_df.to_csv(output_path, index=False)
            
            # Save to database
            db_manager = DatabaseManager()
            if db_manager.connect():
                table_name = f"scheduled_{os.path.splitext(os.path.basename(file_path))[0]}"
                db_manager.save_data_to_table(cleaned_df, table_name)
            
            # Send notification
            notifier = EmailNotifier()
            job_data = {
                'filename': os.path.basename(file_path),
                'original_rows': len(df),
                'processed_rows': len(cleaned_df),
                'processing_time': 0,  # Would track actual time
                'cleaning_report': cleaner.cleaning_report
            }
            notifier.send_processing_complete_email(job_data)
            
            logger.info(f"Scheduled ETL completed for: {file_path}")
            
        except Exception as e:
            logger.error(f"Error in scheduled ETL: {e}")

# Initialize components
db_manager = DatabaseManager()
visualizer = DataVisualizer()
schedule_manager = ScheduleManager()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the file
            start_time = time.time()
            
            # Read file based on extension
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                flash('Unsupported file format')
                return redirect(url_for('index'))
            
            # Clean the data
            cleaner = DataCleaner()
            cleaned_df = cleaner.clean_dataset(df)
            
            # Generate visualizations
            plots = visualizer.generate_comprehensive_report(cleaned_df, filename)
            
            # Save processed data
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cleaned_df.to_csv(processed_filepath, index=False)
            
            # Save to database
            if db_manager.connect():
                table_name = os.path.splitext(filename)[0]
                db_manager.save_data_to_table(cleaned_df, table_name)
                
                # Save job details
                processing_time = time.time() - start_time
                job_data = {
                    'filename': filename,
                    'original_rows': len(df),
                    'processed_rows': len(cleaned_df),
                    'processing_time': processing_time,
                    'status': 'completed',
                    'cleaning_report': cleaner.cleaning_report
                }
                db_manager.save_processing_job(job_data)
            
            # Generate HTML table
            table_html = cleaned_df.head(100).to_html(classes='table table-striped', table_id='data-table')
            
            return render_template('result.html', 
                                 table=table_html,
                                 cleaning_report=cleaner.cleaning_report,
                                 plots=plots,
                                 filename=filename)
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))

@app.route('/schedule')
def schedule_page():
    return render_template('schedule.html')

@app.route('/schedule', methods=['POST'])
def create_schedule():
    job_name = request.form['job_name']
    file_path = request.form['file_path']
    schedule_pattern = request.form['schedule_pattern']
    
    if schedule_manager.schedule_job(job_name, file_path, schedule_pattern):
        flash('Job scheduled successfully!')
    else:
        flash('Error scheduling job!')
    
    return redirect(url_for('schedule_page'))

@app.route('/api/jobs')
def get_jobs():
    """API endpoint to get processing jobs"""
    if db_manager.connect():
        try:
            cursor = db_manager.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT 10")
            jobs = cursor.fetchall()
            return jsonify(jobs)
        except:
            return jsonify([])
    return jsonify([])

@app.route('/api/stats')
def get_stats():
    """API endpoint to get processing statistics"""
    stats = {
        'total_jobs': 0,
        'total_rows_processed': 0,
        'average_processing_time': 0,
        'success_rate': 100
    }
    
    if db_manager.connect():
        try:
            cursor = db_manager.connection.cursor()
            cursor.execute("SELECT COUNT(*) as total, SUM(processed_rows) as total_rows, AVG(processing_time) as avg_time FROM processing_jobs")
            result = cursor.fetchone()
            if result:
                stats['total_jobs'] = result[0]
                stats['total_rows_processed'] = result[1] or 0
                stats['average_processing_time'] = round(result[2] or 0, 2)
        except:
            pass
    
    return jsonify(stats)

if __name__ == '__main__':
    # Initialize database
    if db_manager.connect():
        db_manager.create_tables()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)