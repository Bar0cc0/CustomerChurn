# Customer Churn Analysis

## Overview
A comprehensive data pipeline for analyzing customer churn in telecommunications data, transforming raw customer information into actionable business intelligence. This project integrates SQL Server, Python, scikit-learn, and visualization tools to help businesses understand why customers leave and develop targeted retention strategies.

## Features
- Normalized database schema (3NF) for efficient data storage and querying
- Data lakehouse architecture combining flexibility with structured querying
- Customer segmentation using K-means clustering
- Multi-dimensional cohort analysis (demographics, contracts, services, payment methods, satisfaction)
- Custom visualization framework for consistent reporting
- Automated ETL processes with robust error handling
- Geographic and demographic pattern identification
- Service adoption impact analysis
- SQL Server integration with Python analytics pipeline
- Automated database deployment with PowerShell

## Installation (Windows)
(SQLServer must be installed)
- Create a virtual environment and have it activated
- Run InstallCustomerChurnDatabase.ps1 with appropriate parameters:
  ```powershell
  .\InstallCustomerChurnDatabase.ps1 -ServerInstance $env:COMPUTERNAME
  ```
- To force reinstallation by dropping existing database, add the `-Force` parameter:
  ```powershell
  .\InstallCustomerChurnDatabase.ps1 -ServerInstance $env:COMPUTERNAME -Force
  ```
- The installation script runs the necessary SQL scripts and Python code to create the database, import data, and generate analysis reports
- Analysis reports and visualizations will be available in the Reports directory

## Requirements
- Windows with SQL Server installed
- Python 3.10 or higher
- PowerShell 5.1 or higher
- SQL Server Management Objects
- Required Python packages (installed automatically by the script)