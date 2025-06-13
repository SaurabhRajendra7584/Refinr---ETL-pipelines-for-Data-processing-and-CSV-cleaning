
<<<<<<< HEAD
=======
# 🔥 ETL Pipeline for Data Cleaning and CSV Cleaning Automation Web App

#### 👑 Developed & Owned by: **Saurabh Rajendra**

---

## 🚀 Project Overview

This is a **complete ETL (Extract - Transform - Load) Pipeline Automation Project** built entirely from scratch using Python, Flask, Pandas, MySQL, and advanced automation techniques.

It allows users to:

- Upload raw CSV data
- Automatically clean & transform data
- Load into MySQL database
- Perform analysis & generate plots
- Automate scheduled ETL tasks
- Fully production-ready for deployment

---

## 🧰 Tech Stack

| Layer        | Technology |
| ------------ | ----------- |
| Backend      | Python 3.x, Flask |
| ETL Processing | Pandas, NumPy |
| Database     | MySQL |
| Visualization | Seaborn, Matplotlib, Plotly |
| Scheduler    | APScheduler |
| Deployment   | Render, Railway, Docker |
| Others       | SQLAlchemy, schedule |

---

## 📂 Project Structure

```bash
project/
│
├── app.py                # Main Flask application
├── etl.py                # ETL processing module
├── scheduler.py          # ETL scheduler
├── db_config.py          # MySQL Database configuration
├── requirements.txt      # Python dependencies
├── Procfile              # Deployment configuration
├── logs/                 # ETL logs
├── uploads/              # Raw uploaded files
├── processed/            # Cleaned files
└── templates/
    └── index.html        # Flask HTML template
````

---

## 🔧 Project Setup

### Step 1️⃣ — Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/etl-pipeline-app.git
cd etl-pipeline-app
```

### Step 2️⃣ — Setup Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3️⃣ — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4️⃣ — Install & Setup MySQL Server

* ✅ You can download MySQL Workbench
* ✅ OR install MySQL Server directly (Workbench is optional)

### Step 5️⃣ — Create MySQL Database

Inside MySQL:

```sql
CREATE DATABASE etl_database;
USE etl_database;

CREATE TABLE cleaned_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    col1 VARCHAR(255),
    col2 VARCHAR(255),
    col3 VARCHAR(255)
);
```

Modify this according to your CSV structure.

### Step 6️⃣ — Update MySQL Credentials

Edit `db_config.py`:

```python
DB_CONFIG = {
    'user': 'YOUR_USERNAME',
    'password': 'YOUR_PASSWORD',
    'host': 'localhost',
    'database': 'etl_database'
}
```

---

## 🔁 Application Flow

### ✅ Web Interface Upload

* Open `http://127.0.0.1:5000/`
* Upload raw CSV file via Web Form

### ✅ Automated ETL

* File is automatically cleaned (handled missing values, converted datatypes)
* Cleaned file stored in `/processed/` directory

### ✅ Load into MySQL

* After cleaning, data is automatically inserted into MySQL

### ✅ Visualization

* Various plots generated using Seaborn, Matplotlib, and Plotly (in ETL pipeline)

### ✅ Scheduling (Full Automation)

We added **APScheduler** to schedule the ETL job automatically at fixed intervals.

Example (in `scheduler.py`):

```python
scheduler.add_job(run_etl, 'interval', minutes=5)
```

This means ETL will automatically run every 5 minutes.

---

## ⚙ Deployment Ready

This app can be easily deployed on:

* **Render** ✅
* **Railway** ✅
* **Docker** ✅
* **Vercel (Backend API + Frontend)** ✅
* **VPS Server** ✅

Includes:

* `requirements.txt`
* `Procfile`
* Clean folder structure for CI/CD pipelines

---

## 🔒 Security

* File validation
* Error handling & exception management
* Clean directory separation for raw, processed, and logs

---

## 🌟 Future Enhancements

* User Authentication system (Admin Panel)
* Multi-format file support (XLSX, JSON, Parquet)
* Real-time monitoring dashboard
* Email Notifications after ETL job completion
* Full ML-Driven Data Cleaning Pipeline

---

## 👑 Developed By

> **Saurabh Rajendra**

* 🔗 GitHub: [github.com/YOUR\_GITHUB\_USERNAME](https://github.com/SaurabhRajendra7584)
* 🔗 LinkedIn: (https://www.linkedin.com/in/saurabhrajendradubey)

---

## 🙏 Special Notes

> This entire project was developed as part of my personal journey to master **End-to-End Data Engineering**
> including Web App, ETL, Automation, Database Integration & Deployment.

---

## 📸 Screenshots

*(You may add screenshots of your Web App UI, upload screen, ETL logs, and MySQL tables here for more impact)*
![image](https://github.com/user-attachments/assets/68a59b79-959d-45a3-97fa-0a22a029168b)
![image](https://github.com/user-attachments/assets/928f82bb-6561-456e-af89-94c43d0d4b5a)
![image](https://github.com/user-attachments/assets/22fdbb25-5e82-472d-b0ad-901fa729693a)



## 💻 License

This project is for educational & personal learning purposes and uses MIT License.
>>>>>>> 204c86228f93037c408b188d7e0d39b676df155f
