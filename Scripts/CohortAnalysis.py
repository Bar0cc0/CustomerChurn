#!/usr/bin env python
# -*- coding: utf-8 -*-

"""
Author: Michael Garancher
Date: 2025-03-01
Description: TELCO CUSTOMER CHURN ANALYSIS AND COHORT MODELING

	1. Data Acquisition: Fetch data from the SQLServer database
	2. Data Preprocessing: Clean the data, handle missing values, and standardize numerical features
	3. Exploratory Data Analysis: Check for class imbalance, outliers, and correlation between numerical features
	4. Customer Segmentation: Use KMeans clustering to segment customers into groups
	5. Customer Profiling and Cohort Analysis: Analyze customer demographics, contracts, services, and internet types
	6. Data warehousing and Reporting: Store the results in a database and generate reports

"""

# Import libraries
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt	
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from SQLServerConnector import SQLServerConnector
from CohortModeling import KMeansClustering


# Suppress warnings
sys.tracebacklimit = 0	
warnings.filterwarnings('ignore')

# Set pandas options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 999)
pd.set_option('display.float_format', '{:.2f}'.format)
plt.style.use('ggplot')

# Set the root directory of the project
ROOT:Path = Path(__file__).resolve().parents[1]

# Set the configuration parameters
CONFIG:dict = {
    'input_dir': Path(ROOT).joinpath('Datasets'),
	'output_dirs': {
		'Reports': Path(ROOT).joinpath('Reports'),
		'Models': Path(ROOT).joinpath('Models')
	},
    'db_config': {
        'db_name': 'CustomerChurnDB',
        'schema': 'TelcoChurnQ3'
    }
}

# Report file
reports_dir = CONFIG['output_dirs']['Reports']
reports_dir.mkdir(parents=True, exist_ok=True)
OUT = open(f'{reports_dir}/CohortAnalysis.txt', 'w')



''' HELPER FUNCTIONS '''

# Delegate class for plotting data
class Plot:
	"""
	A utility class for creating and managing matplotlib/seaborn plots.
	This class provides a convenient interface for creating various types of plots
	including histograms, heatmaps, tables, boxplots, and stacked bar charts. It handles 
	subplot management and provides methods for displaying and saving the plots.
	Attributes:
		fig (matplotlib.figure.Figure): The matplotlib figure object.
		ax (list or numpy.ndarray): The subplot axes.
		title (str): The main title of the figure.
		counter (int): Counter for tracking the number of plots created.
		kde (bool): Whether to show KDE in histogram plots.
		stack (bool): Whether to stack bars in bar plots.
		legend_title (str): Title for the legend.
		num_items_stacked (int): Number of items to stack in a stacked bar chart.
		idx (int): Current subplot index.
	Methods:
		getFigSize: Static method to calculate optimal subplot dimensions.
		show: Removes unnecessary axes and renders the plot.
		save: Saves the plot to a file.
	"""

	def __init__(self, figsize:tuple=(16,10), nrows:int=1, ncols:int=1, title:str=None):
		self.fig, self.ax = plt.subplots(nrows, ncols, figsize=figsize)
		self.title = title
		self.fig.suptitle(self.title, fontsize=16) if self.title else None
		self.fig.subplots_adjust(hspace=0.3, wspace=0.3, )
		if not isinstance(self.ax, list):
			self.ax = self.ax.flatten() if nrows > 1 or ncols > 1 else [self.ax] # Flatten axes for 1D indexing
		
	def __call__(self, dataframe:pd.DataFrame, kind:str, idx:int=0, feature:str=None, title:str=None, **kwargs):
		self.idx = idx
		self.counter = kwargs.pop('counter', 0)
		self.kde = kwargs.pop('kde', True)
		self.stack = kwargs.pop('stack', False)
		self.legend_title = kwargs.pop('legend_title', '')
		self.num_items_stacked = kwargs.pop('num_items_stacked', 1)
		self.ax[self.idx].set_xlabel(feature)
		self.ax[self.idx].set_title(title if title else feature)
		
		match kind:
			case 'hist':	
				sns.histplot(dataframe[feature], ax=self.ax[self.idx], kde=self.kde, **kwargs)
				self.ax[self.idx].set_ylabel('Count', fontsize=8)
			case 'heat':
				sns.heatmap(dataframe, ax=self.ax[self.idx], **kwargs)	
			case 'table':			
				self.ax[self.idx].table(
					cellText=dataframe.values, 
					colLabels=dataframe.columns, 
					rowLabels=dataframe.index,
					**kwargs
				).set_fontsize(8)
				self.ax[self.idx].axis('off')
			case 'box':
				self.ax[self.idx].set_ylabel('')
				self.ax[self.idx].set_xlabel('')
				self.ax[idx].set_xlabel('')
			case 'stack':
				stacked_items_boundaries_list = np.zeros(self.num_items_stacked) # NB: Number of clusters
				for cluster_id, val in zip(dataframe.index, dataframe.values):
					l = self.ax[idx].bar(dataframe.columns, val, bottom=stacked_items_boundaries_list, label=cluster_id)
					self.ax[idx].bar_label(l, labels=[f"{v:.1f}%" if v > 0 else '' for v in val], label_type='center', fontsize=6)
					stacked_items_boundaries_list += val
				self.ax[idx].legend(title=self.legend_title,bbox_to_anchor=(0,0.99), loc='upper right', fontsize=8)
				self.ax[idx].get_yaxis().set_visible(False)
			case _:
				raise ValueError(f"Invalid plot type: {kind}. Choose from 'hist', 'heat', 'table', 'box'")
	
	@staticmethod
	def getFigSize(n_features:int) -> tuple:
		n_rows = int(np.ceil(np.sqrt(n_features)))
		n_cols = int(np.ceil(n_features / n_rows))
		return (n_rows, n_cols)

	def show(self):
		try:
			# remove empty axes
			if self.counter >= 2 and self.counter < len(self.ax):
				for idx in range(self.counter+1, self.ax.shape[0]):
					self.ax[idx].axis('off')
			# Apply formatting to the axes
			for ax in self.ax:
				ax.tick_params(axis='both', labelsize=8)
				ax.set_title(ax.get_title(), fontsize=10)
				ax.set_xlabel(ax.get_xlabel(), fontsize=8)
				ax.set_ylabel(ax.get_ylabel(), fontsize=8)
				
				
		except AttributeError:...
		finally:
			plt.tight_layout()

	def save(self, filename:str):
		filename = filename if filename.endswith('.png') else filename + '.png'
		plots_dir = CONFIG['output_dirs']['Reports'] / 'Plots'
		plots_dir.mkdir(parents=True, exist_ok=True)
		try:
			self.fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {plots_dir / filename}")
		except Exception as e:
			print(f"Error saving figure: {e}")
	
# Group numerical features into bins of size 'bin_size'
def group_into_bins(dfcol:pd.DataFrame, bin_size:int, **kwargs)->pd.Series:
	_min = kwargs.pop('_min', (np.floor(dfcol.min()/bin_size)*bin_size).astype(int))
	_max = kwargs.pop('_max', (np.ceil(dfcol.max()/bin_size)*bin_size).astype(int))
	bins = range(_min, _max+1, bin_size)
	labels = [f'{i}-{i+bin_size-1}' for i in range(_min, _max, bin_size)]
	return pd.cut(dfcol, bins=bins, labels=labels, include_lowest=True)

def create_percentage_crosstab(df, index_col, column_col, normalize='index'):
    """Create a crosstab with percentage values"""
    result = pd.crosstab(df[index_col], df[column_col], normalize=normalize) * 100
    return result.round(2)

def safe_db_operation(operation_func):
    """Decorator for safe database operations with proper error handling"""
    def wrapper(*args, **kwargs):
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            print(f"Database operation failed: {e}")
            #TODO Log error details
            #TODO Attempt rollback if applicable
            return None
    return wrapper



''' 1. DATA ACQUISITION '''

# Fetch data from server into a DataFrame object
connector	= SQLServerConnector(CONFIG['db_config']['db_name'])
conn 		= connector.connect()
query 		= "SELECT * FROM Lakehouse.Customers"
df 			= connector.fetch_data(query)
connector.close()

# Keep a copy of the original dataset
df_original_copy = df.copy(deep=True) 




''' 2. DATA PREPROCESSING '''

# Check for missing values in both sources (i.e., SQLServer and CSV)
def crosscheck_missing_values(df: pd.DataFrame, csv:Path) -> None:
	# Clean missing values and format column names
	df = df.replace( [_ for _ in ['NULL','NA','None']], pd.NA) 
	df2 = pd.read_csv(csv)
	df2.columns = df2.columns.str.replace(' ', '_')
	# Check for missing values in both sources
	count_comparison = [df.isnull().sum(), df2.isnull().sum()]
	count_comparison_table = pd.concat(count_comparison, axis=1, keys=['SQLServer', 'CSV'])\
							   .where(lambda x: x>0).dropna().astype(int)
	# Visualize missing values
	cmv_plot = Plot(title='Missing Values in the Telco Customers Dataset', ncols=2)
	cmv_plot(
		df.isnull(), 'heat', 0, 
		'SQLServer', title='SQLServer: Rows with Missing Values per Column', 
		cbar=False
	)
	cmv_plot(
		count_comparison_table, 'table', 1, 
		title='Counts of Missing Values in SQLServer and CSV sources',
		loc='best', colWidths=[0.3, 0.3]
	)
	cmv_plot.show()
	cmv_plot.save('DESC_Missing_Values.png')
crosscheck_missing_values(df, CONFIG['input_dir'] / 'telco_customers.csv')

def check_duplicates(df:pd.DataFrame) -> None:
	# Check for duplicate rows
	if (s := df.duplicated().sum()) > 0:
		print(f"## Number of duplicate rows: {s}", end='\n\n', file=OUT)
		df.drop_duplicates(inplace=True)

	# Check for duplicate columns	
	if s := df.columns.duplicated().sum() > 0:
		print(f"## Number of duplicate columns: {s}", end='\n\n', file=OUT)
		df = df.loc[:, ~df.columns.duplicated()]


# Descriptive statistics
cat_features  = df.describe(include=['object']).drop(['count'])
num_features  = df.describe(include=['float64', 'int64']).drop(['count'])
unique_values = df.describe(include='all').loc['unique'].where(lambda x: x>1).dropna()
print(
	'TELCO CUSTOMERS DATASET | Q3 2025',
	'---------------------------------\n',
	'#'*23 + '\n# DATASET DESCRIPTION #\n'+'#'*23 ,
	f'## Total number of rows: {df.shape[0]}',
	f'## Total number of features: {df.shape[1]}',
	f'## Duplicated rows/columns: {check_duplicates(df)}',
	f'## Categorical features: {df.columns[df.dtypes == "object"].to_list()}', 
	f'## Numerical features: {df.columns[df.dtypes != "object"].to_list()}',
	f'## Features with unique values: {unique_values.index.to_list()}',
	"## Descriptive statistics for the 'Churned' customers",
	df.drop(columns=['Latitude', 'Longitude', 'Zip_Code'])\
		.groupby('Customer_Status').get_group('Churned')\
		.rename(columns={
			'Number_of_Dependents': 'n_Dependents',
			'Number_of_Referrals': 'n_Referrals',
			'Tenure_in_Months': 'Tenure(mm)',
			'Total_Extra_Data_Charges': 'Xtra_Data($)',
			'Total_Long_Distance_Charges': 'Long_Dist($)',
			'Avg_Monthly_Long_Distance_Charges': 'Avg_Long_Dist($/mm)',
			'Avg_Monthly_GB': 'Avg_Data(GB/mm)',
			'Satisfaction_Score': 'Satisfaction',
		})\
		.describe()[2:].round(2),
	sep='\n\n', end='\n\n', flush=True, file=OUT
)


# Cast Zip_Code to string to avoid it being treated as a numerical feature
df['Zip_Code'] = df['Zip_Code'].astype('str')

# Remove columns with only one unique value
df = df.loc[:, df.nunique() > 1]

# Remove categorical features with high cardinality
high_cardinality_cols = df.select_dtypes('object').nunique()\
						  .where(lambda x: x>df.shape[0]//4).dropna()\
						  .index.to_list()
df_card = df[high_cardinality_cols]
df = df.drop(columns=high_cardinality_cols)

# Store a list of features of int type before converting binary features to boolean
native_int_cols = df.select_dtypes(include=int).columns.to_list()

# Convert binary features to boolean
binary_features = unique_values.where(lambda x: x==2).dropna().index.to_list()
df[binary_features] = df[binary_features].applymap(lambda x: 1 if x in ['Yes', 'Male'] else 0)
int_cols = df.select_dtypes(include=int).columns.to_list()
print(f"# Features converted to boolean: {df[binary_features].columns.to_list()}", end='\n\n', file=OUT)

# Standardize data distribution of numerical features
float_cols = df.select_dtypes(include=[float]).columns.to_list()
[float_cols.remove(_) for _ in ['Latitude', 'Longitude']]
df[float_cols] = StandardScaler().fit_transform(df[float_cols])

# All numerical features
num_cols = int_cols + float_cols




''' 3. EXPLORATORY DATA ANALYSIS '''

print(
	"\n"+'#'*29 + '\n# EXPLORATORY DATA ANALYSIS #\n' + '#'*29,
	sep='\n\n', end='\n\n', file=OUT
)

# Check for Churn Categories and Reasons
def check_churn_categories() -> None:
	churn_reasons = df['Churn_Reason'].value_counts()
	churn_categories = df['Churn_Category'].value_counts()
	print(
		"## Churn Categories and Reasons",
		f"### Churn Categories:\n{churn_categories}",
		f"### Churn Reasons:\n{churn_reasons}",
		sep='\n\n', end='\n\n', file=OUT
	)
check_churn_categories()

# Check churn rate and retention rate
def check_churn_rate() -> None:
	n_records 	= df.shape[0]
	n_churned 	= df['Customer_Status'].value_counts().loc['Churned']
	n_joined	= df['Customer_Status'].value_counts().loc['Joined']
	n_stayed	= df['Customer_Status'].value_counts().loc['Stayed']
	churn_rate 	= ((n_churned / (n_records-n_joined)) * 100).round(2)
	retention_rate = ((n_stayed / (n_records-n_joined)) * 100).round(2)

	print(
		f'# Churn rate: {churn_rate}%',
		f'# Retention rate: {retention_rate}%',
		sep='\n\n', end='\n\n', file=OUT
	)
check_churn_rate()

# Check for correlation between numerical features: Charges and Scores
def check_corr() -> None:
	charges_df = df[['Avg_Monthly_Long_Distance_Charges', 'Monthly_Charge', 'Total_Charges', 'Total_Extra_Data_Charges', 'Total_Long_Distance_Charges', 'Total_Revenue']]
	charges_corr_matrix = charges_df.corr(method='pearson', numeric_only=True)
	scores_df = pd.DataFrame(
		StandardScaler().fit_transform(df[['Satisfaction_Score', 'Churn_Score', 'CLTV']]), 
		columns=['Satisfaction_Score', 'Churn_Score', 'CLTV']
	)
	scores_corr_matrix = scores_df.corr(method='pearson')
	cc_plot = Plot(title='Charges and Scores', ncols=2)
	cc_plot(charges_corr_matrix.round(2), 'heat', 0, title='Correlation between Charges', annot=True, center=0)
	cc_plot(scores_corr_matrix.round(2), 'heat', 1, title='Correlation between Scores', annot=True, center=0)
	cc_plot.show()
	cc_plot.save('EDA_Correlation_Charges_Scores.png')
check_corr()

# Check for distribution in numerical features
def check_numeric_features() -> None:
	n_rows, n_cols = Plot.getFigSize(n_features=len(num_cols))
	distribution_plot = Plot(figsize=(15, 10), 
						  	 nrows=n_rows, 
						  	 ncols=n_cols, 
						  	 title='Distribution of Numerical Features in the Telco Customers Dataset'
	)
	counter:int = 0
	for idx, feat in enumerate(num_cols):
		distribution_plot(df[feat].to_frame(), 'hist', idx, feat, counter=counter)
		counter += 1
	distribution_plot.fig.text( x=0.8, y=0.1, 
								s='Binary features:\n\n0: {No, Female}\n1: {Yes, Male}', 
								fontsize=10, verticalalignment='center'
	)
	distribution_plot.show()
	distribution_plot.save('EDA_Distribution_Numerical_Features.png')
check_numeric_features()

# Check for outliers in numerical features
def check_outliers() -> None:
	n = float_cols + native_int_cols
	n_rows, n_cols = Plot.getFigSize(n_features=len(n))
	outliers_plot = Plot(figsize=(15, 10), 
                         nrows=n_rows,
                         ncols=n_cols, 
                         title='Outliers in numerical features'
	)
	counter:int = 0
	for idx, feat in enumerate(n):
		outliers_plot(df[feat].to_frame(), 'box', idx, feat, counter=counter)
		counter += 1
	outliers_plot.show()	
	outliers_plot.save('EDA_Outliers_Numerical_Features.png')
check_outliers()

# Check for class imbalance in the target feature (Customer_Status)
def check_class_imbalance() -> None:
	class_imbalance_plot = Plot(figsize=(10, 7), 
							 	ncols=1, 
								title='Class Imbalance in the taget feature Customer_Status'
	)
	s = df['Customer_Status'].value_counts(normalize=True)
	s = pd.DataFrame(s.apply(lambda x: x*100).values, index=s.index, columns=['Percentage'])
	sns.countplot(x='Customer_Status', data=df, ax=class_imbalance_plot.ax[0])
	class_imbalance_plot.ax[0].set_yticklabels([f"{x}%" for x in range(0, 101, 10)], fontsize=8)
	class_imbalance_plot.ax[0].set_ylabel('Percentage', fontsize=8)
	class_imbalance_plot.show()
	class_imbalance_plot.save('EDA_Class_Imbalance_Customer_Status.png')
check_class_imbalance()






''' 4. CUSTOMER SEGMENTATION '''

# Create a KMeans model to segment customers
model = KMeansClustering(df, 'Churn_Score') 

""" BEGIN: COMMENT ONCE THE MODEL IS TRAINED """

# Fit the model, predict clusters, evaluate the model, and visualize the clusters
df_with_clusters = model.fit_predict()
model.evaluate()
model.visualize_clusters(
	df_with_clusters, 
	save_path=Path(CONFIG['output_dirs']['Reports']).joinpath('Plots/kmeans_plot.png')
)
# Save the model for future use
model.save_model(Path(CONFIG['output_dirs']['Models']).joinpath('kmeans_model.pkl'))

""" END: COMMENT ONCE THE MODEL IS TRAINED """

# Once the model is saved, it can be loaded and used to predict clusters on new data
#TODO: Change the path to the model file
model.load_model(Path(ROOT).joinpath('Models/kmeans_model.pkl'))
df_with_clusters = model.predict()

# Explain the clusters
model.explain_clusters(df_with_clusters)

# Get descriptive statistics and visualizations for the clusters
def get_cluster_stats(df:pd.DataFrame) -> pd.DataFrame:	
	"""
	Compute descriptive statistics and visualizations for the clusters.
	1. Cluster statistics (count, churn rate, churn score, CLTV, tenure)
	2. Correlation between CLTV, Churn_Score, and Cluster
	3. Churn_Score threshold
	4. Churn rate wrt Clusters
	5. Clusters, Churn Score and CLTV wrt Customer Status
	"""
	# ===== 1. COMPUTE DESCRIPTIVE STATS =====

	# 1.1 Cluster statistics
	
	# Group Churn_Score into bins of size 20
	df['Churn_Score_Bin'] = group_into_bins(df['Churn_Score'],20)
	df['CLTV_Bin'] = group_into_bins(df['CLTV'], 1000, _min=0)
	
	# Clusters statistics uni- and bivariate
	cluster_stats = df.groupby('Cluster').agg({
		'Cluster': ['count'],
		'Customer_Status': [lambda x: (x=='Churned').mean() * 100],
		'Churn_Score': ['mean'],
		'CLTV': ['mean'],
		'Tenure_in_Months': ['mean'],
	})
	cluster_stats.columns = ['Rows Count', 'Churn %', 'Churn Score Avg', 'CLTV Avg', 'Tenure M Avg']
	cluster_stats = cluster_stats.sort_values('Churn %', ascending=False)

	# 1.2 Correlation between CLTV, Churn_Score, and Cluster
	cluster_corr = df[['CLTV', 'Churn_Score', 'Cluster']].dropna().corr()
	print(
		"\n## Cluster 1: Correlation to target features matrix", 
		cluster_corr, 
		sep='\n', end='\n\n', file=OUT
	)
	
	# 1.3 Churn_Score threshold
	s = df.copy(deep=True)\
			.groupby('Churn_Score_Bin')['Customer_Status']\
			.value_counts().unstack()\
			.apply(lambda x: x['Churned'] / (x['Churned'] + x['Stayed']), axis=1)\
			.sort_values(ascending=False)\
			.rename_axis('Churn Rate')
	threshold = s.where(lambda x: x >= 0.3).dropna()
	
	# 1.4 Churn rate wrt Clusters
	churn_wrt_cluster = df.groupby('Cluster')['Customer_Status'].value_counts().unstack().fillna(0).astype(int)
	churn_wrt_cluster['Churn Rate'] = churn_wrt_cluster['Churned'] / (churn_wrt_cluster['Churned'] + churn_wrt_cluster['Stayed']) * 100
	churn_wrt_cluster = churn_wrt_cluster.sort_values('Churn Rate', ascending=False).rename(columns={'Churn Rate':'Churn Rate %'})
	
	# 1.5 Clusters, Churn Score and CLTV wrt Customer Status
	a=churn_wrt_cluster.sort_index()
	b=df.copy(deep=True).groupby('Customer_Status')['Churn_Score_Bin'].value_counts().unstack().fillna(0).T
	c=df.copy(deep=True).groupby('Customer_Status')['CLTV_Bin'].value_counts().unstack().fillna(0).T
	
	# 1.6 Print cluster statistics summary
	print(
		f"\n## Top 2 Churn Score Bins with highest churn rate (i.e., >.3):\n{threshold}",
		"\n## Clusters Description Univariate", 
		f"{a}\n=> Cluster 1 accounts for {(a['Churned'].loc[1] / a['Churned'].sum() * 100).round(2)}% of the Churned customers",
		"\n## Cluster 1 w.r.t. Customer_Status: Churn_Score and CLTV",
		f"{b}\n=> Churn_Score > 60 accounts for {(b['Churned'].iloc[3:4].sum() / b['Churned'].sum() * 100).round(0)}% of the Churned customers",
		f"{c}\n=> CLTV 2000<>6000 accounts for {(c['Churned'].iloc[2:5].sum() / c['Churned'].sum() * 100).round(2)}% of the Churned customers",
		sep="\n\n", end="\n\n", flush=True, file=OUT
	)

	# ===== 2. VISUALIZATIONS =====
	# Churn rate wrt Clusters
	churn_wrt_cluster.drop(columns='Churn Rate %', inplace=True)
	status_plot = Plot()
	churn_wrt_cluster.plot(
		kind='bar', stacked=True, ax=status_plot.ax[0], rot=0.0,
		title='Distribution of Customer Status by Cluster'
	)
	status_plot.ax[0].set_ylabel('Count')
	status_plot.show()
	status_plot.save('CA_Status_by_Cluster.png')

	# Pairplot of Churn_Score, CLTV, and Cluster
	pair_plot = sns.pairplot(
		df, 
		vars=['Churn_Score', 'CLTV', 'Cluster'], 
		hue='Customer_Status',
		markers=['o', 's', 'D'],
	)
	pair_plot.figure.suptitle('Pairplot of Churn Score, CLTV, and Cluster')
	pair_plot.figure.subplots_adjust(top=0.95)
	pair_plot.savefig(Path(ROOT).joinpath('Reports/Plots/CA_Corr_to_Targets.png'))

	return cluster_stats
get_cluster_stats(df_with_clusters)







''' 5. CUSTOMER PROFILING AND COHORT ANALYSIS '''


# Group Age into bins of size 10 years
df_with_clusters['Age_Bin'] = group_into_bins(df['Age'], 10)

# Group Population into bins of size 10000 habitants
df_with_clusters['Population_Bin'] = group_into_bins(df['Population'], 10000, _min=0)

# Group Monthly_Charge into bins of size 10$
df_with_clusters['Monthly_Charge_Bin'] = group_into_bins(df['Monthly_Charge'], 10)

# Group Tenure_in_Months into bins of size 10 months
df_with_clusters['Tenure_Bin'] = group_into_bins(df['Tenure_in_Months'], 10)



## Demographics Analysis
def DemographicsAnalysis(df:pd.DataFrame) -> None:
	"""
	Analyzes demographic characteristics of customer clusters with a focus on Cluster 1.
		
	Business Context:
	- Customer demographics strongly influence product preferences and churn risk
	- Geographic concentration of high-risk customers enables targeted regional campaigns
	- Family status (marriage, dependents) affects service bundle decisions and price sensitivity
	- Age distribution reveals potential generational technology adoption barriers
	- Understanding demographic patterns allows for more personalized retention strategies

	Analysis Goals:
	1. Identify demographic factors most predictive of churn risk
	2. Discover geographic hotspots of high-risk customers for localized campaigns
	3. Determine family composition patterns associated with service adoption/cancellation
	4. Develop demographic profiles for each customer segment
	5. Support targeted marketing initiatives and product development decisions

	Key Methods:
	- Demographic attribute distribution across clusters (gender, age, marital status)
	- Geographic analysis of high-risk customers by city and population
	- Household composition analysis (dependents, family size)
	- Identification of top 10 cities with highest concentration of at-risk customers
	"""
	print(
		"\n"+"#"*25,
		"# DEMOGRAPHICS ANALYSIS #", 
		"#"*25,
		sep="\n", end="\n\n" ,file=OUT
	)

	# ===== 1. DEMOGRAPHICS =====
	
	# 1.1 Demographics by cluster
	stats_dict = {
		'Gender': ['mean'],
		'Age': [lambda x: x.median().astype(int)],
		'Married': [lambda x: (x==1).mean() * 100],
		'Number_of_Dependents': ['mean'],
		'Number_of_Referrals': ['mean'],
		'Population': [lambda x: x.median().astype(int)],
	}
	columns = ['Gender Ratio', 'Age Median', 'Married %', 'Dependents Avg', 'Referrals Avg', 'Population Median']
	group = df.groupby(['Cluster'])
	overview = group.agg(stats_dict)
	overview.columns = columns
	print(
		f"## Overview\n{overview}", 
		sep="\n\n", end='\n\n', file=OUT
	)

	# ===== 2. CLUSTER 1 CUSTOMER PROFILE =====

	# 2.1 Population distribution 
	c1_city = group.get_group(1)\
					.groupby('Population')[['City']].value_counts().sort_values(ascending=False)\
					.groupby('City').sum().sort_values(ascending=False)
	top_10_cities = c1_city.nlargest(10)							
	
	print(
		"## Cluster 1: Population Distribution",
		f"=> Top 10 cities with the highest population account for {top_10_cities.sum() / c1_city.sum() * 100:.2f}% of the customers in Cluster 1",
		sep='\n\n', end='\n\n', file=OUT
	)

	# 2.3 Marital status and Dependents distribution 
	c1_marital = group.get_group(1)\
						.groupby('City')['Married'].mean() * 100
	c1_dependents = group.get_group(1)\
						.groupby('City')['Number_of_Dependents'].mean()
	
	
	# 2.4 Age distribution
	c1_age = group.get_group(1)\
					.groupby('City')[['Under_30', 'Senior_Citizen']].mean() * 100
						
	top_10_joined = c1_city.to_frame()\
							.join(c1_marital.to_frame())\
							.join(c1_dependents.to_frame())\
							.join(c1_age)
	top_10_joined.rename(
		columns={
			'count': 'Customers Count', 
			'Married':'Married %', 
			'Number_of_Dependents':'Household Size Avg',
			'Under_30':'under 30y.o. %', 
			'Senior_Citizen':f"over {df.query( 'Senior_Citizen == 1' )['Age'].min()}y.o. %"}, 
		inplace=True
		)
	print(
		top_10_joined.head(10),
		sep='\n\n', end='\n\n',	file=OUT
	)  			

	# ===== 3. VISUALIZATIONS =====

	Demographics_plot = Plot(nrows=2, ncols=2, title='Cluster 1 Demographics')
	Demographics_plot.fig.subplots_adjust(hspace=0.5, wspace=0.3)
	
	top_10_cities.plot(kind='bar', ax=Demographics_plot.ax[0])
	Demographics_plot.ax[0].set_xticklabels(top_10_cities.index, rotation=45, fontsize=6)
	Demographics_plot.ax[0].set_xlabel('City', fontsize=8)
	Demographics_plot.ax[0].set_title('Top 10 Cities by Customers Count', fontsize=10)
	
	Demographics_plot(
		group.get_group(1), 'hist', 1, 
		feature='Age', bins=group.get_group(1)['Age'].nunique(),
		title='Age Distribution', kde=False
	)
	
	Demographics_plot(
		group.get_group(1), 'hist', 2, 
		feature='Married', bins=2, 
		title='Marital Status Distribution', kde=False
	)
	Demographics_plot.ax[2].set_xticks([0.25, 0.75])
	Demographics_plot.ax[2].set_xticklabels(['Not Married', 'Married'], fontsize=8)
	
	b = group.get_group(1)['Number_of_Dependents'].nunique()
	Demographics_plot(
		group.get_group(1), 'hist', 3, 
		feature='Number_of_Dependents', bins=b, 
		title='Number of Dependents Distribution', kde=False)	
	Demographics_plot.ax[3].set_xticks(range(b))
	Demographics_plot.ax[3].set_xticklabels(range(b), fontsize=8)

	Demographics_plot.save('CA_Demographics.png')
DemographicsAnalysis(df_with_clusters)

## Contract and Services Analysis
def ContractServicesAnalysis(df: pd.DataFrame) -> None:
	"""
    Analyze contract types and services for customers across clusters.
    
    Business Context:
    - Contract type is one of the strongest predictors of churn risk
    - Service combinations reveal opportunities for strategic bundling
    - Internet service quality/type directly impacts customer experience
    - Understanding which services are valued by stable customers guides product development
    - Identifying high-churn service combinations pinpoints potential quality or pricing issues
    
    Analysis Goals:
    1. Quantify the retention impact of different contract types across segments
    2. Evaluate promotional offer effectiveness for different customer segments
    3. Identify service combinations with highest revenue stability
    4. Discover "gateway services" that lead to higher long-term customer value
    5. Support contract redesign and upsell/cross-sell opportunity identification
    
    Key Methods:
    - Contract distribution analysis across customer segments
    - Service adoption patterns and correlation with revenue and retention
    - Internet service type performance analysis
    - High-value service combination identification
    - Churn rate analysis by contract type and service bundle
	"""
	print(
		"\n"+"#"*33,
		"# CONTRACT AND SERVICE ANALYSIS #", 
		"#"*33,
		sep="\n", end="\n\n" ,file=OUT
	)
	
    # ===== 1. CONTRACT ANALYSIS =====
	print("\n## CONTRACT AND OFFER", file=OUT)

	# 1.1 Contract and Offer distribution by cluster
	group = df.groupby('Cluster')
	contract_overview = group[['Contract']].value_counts(normalize=True).unstack().fillna(0) * 100
	offer_overview = group[['Offer']].value_counts(normalize=True).unstack().fillna(0) * 100

	print(
		f"### Contract Distribution by cluster (%):\n{contract_overview}", 
		f"### Offer Distribution by cluster (%):\n{offer_overview}",
		sep='\n\n',	end='\n\n', file=OUT
	)
	
	# 1.2 Contract impact on churn and revenue
	contract_impact = df.groupby(['Cluster', 'Contract']).agg({
        'Customer_Status': lambda x: (x == 'Churned').mean() * 100,  
        'Monthly_Charge': 'mean',                                   
        'Tenure_in_Months': 'mean',                              
        'Total_Revenue': 'mean'                                   
    }).round(2)    
	print(f"### Contract Impact on Churn and Revenue\n{contract_impact}", file=OUT)

    # 2.2 Offer effectiveness (churn rate, revenue, tenure)
	offer_effect = df.groupby(['Cluster', 'Offer']).agg({
		'Customer_Status': lambda x: (x == 'Churned').mean() * 100,
		'Monthly_Charge': 'mean',
		'Tenure_in_Months': 'mean',
		'CLTV': 'mean'
	}).round(2)
	print(f"\n### Offer Effectiveness:\n{offer_effect}", file=OUT)



	# ===== 2. INTERNET AND SERVICES ANALYSIS =====
	print("\n## INTERNET AND SERVICES", file=OUT)
	
	service_columns = [col for col in df.columns if any(
        service in col.lower() for service in ['phone', 'internet', 'security', 'backup', 
                                             'protection', 'tech', 'stream', 'unlimited']
    )]
	
	df = df.reset_index().rename(columns={'index': 'row_id'})
	id_col = 'row_id'

	# 2.1 Internet type
	cat_services = [col for col in service_columns if pd.api.types.is_object_dtype(df[col])]
	cat_melted = pd.melt(
		df,
		id_vars=[id_col, 'Cluster'], 
		value_vars=cat_services,
		var_name='Service',
		value_name='Value'
	)
	for service in cat_services:
		service_data = cat_melted[cat_melted['Service'] == service]
		service_pivot = pd.pivot_table(
			service_data,
			index='Value',
			columns='Cluster',
			values=id_col,
			aggfunc='count',
			fill_value=0
		)
		internet_type = service_pivot.div(service_pivot.sum(axis=0), axis=1) * 100
		print(f"\n### {service} Adoption by Cluster (%):\n{internet_type}", sep='\n', end='\n\n', file=OUT)
	
	# 2.2 Services adoption
	bin_services = [col for col in service_columns if not pd.api.types.is_object_dtype(df[col])]
	services_adoption = (df.groupby('Cluster')[bin_services].mean() * 100).T 
	print(f"\n### Services Adoption by Cluster (%)\n{services_adoption.round(2)}", sep='\n', end='\n\n', file=OUT)

	# 2.3 High-value service combinations
	c1 = df[df['Cluster'] == 1]      
	abrv = {
        'Online_Security': 'OS',
        'Online_Backup': 'OB',
        'Device_Protection_Plan': 'DP',
        'Premium_Tech_Support': 'PTS',
        'Streaming_TV': 'STV',
        'Streaming_Movies': 'SM',
        'Streaming_Music': 'SM',
        'Unlimited_Data': 'UD',
        'Multiple_Lines': 'ML',
        'Phone_Service': 'PS',
		'Internet_Service': 'IS'
    }

	
	service_combos = c1[bin_services].apply(lambda row: 
		' + '.join(sorted([col.replace(str(col), abrv[str(col)]) for col, val in row.items() if val == 1])), axis=1)
	top_combos = service_combos.value_counts().head(5)
	print(f"\n### Top Service Combinations in Cluster 1\n{top_combos}", file=OUT)

	# 2.4 Service combination churn rates
	combo_churn = c1.groupby(service_combos)['Customer_Status'].apply(
		lambda x: (x == 'Churned').mean() * 100
	).sort_values(ascending=False)
	print(f"\n### Service Combination Churn Rates in Cluster 1 (%)\n{combo_churn.head().round(2)}", file=OUT)


    # ===== 5. VISUALIZATIONS =====

	Contract_plot = Plot(ncols=2, title='Contracts and Offers Distribution by Cluster')
	Contract_plot(
		contract_overview.T, 'stack', 0, num_items_stacked=5, 
		feature='Cluster', 
		title='Contract Distribution by Cluster', legend_title='Contract'
	)
	
	Contract_plot(
		offer_overview.T, 'stack', 1, num_items_stacked=5, 
		feature='Cluster', 
		title='Offer Distribution by Cluster', legend_title='Offer'
	)
	Contract_plot.save('CA_Contracts.png')

	Services_plot = Plot(figsize=(16, 10), nrows=2, ncols=2, title='Internet Types and Services Distribution by Cluster')
	gs = gridspec.GridSpec(2, 2, Services_plot.fig)
	Services_plot.ax[2] = plt.subplot(gs[1, :])

	Services_plot(
		internet_type, 'stack', 0, num_items_stacked=5, 
		feature='Cluster', 
		title='Internet Type Distribution', legend_title='Type'
	)
	Services_plot(
		services_adoption, 'stack', 1, num_items_stacked=5, 
		feature='Cluster', 
		title='Services Adoption by Cluster w.r.t. Cluster Sizes', legend_title='Service'
	)
	
	top_combos.plot(kind='barh', ax=Services_plot.ax[2])
	Services_plot.ax[2].set_title('Top Service Combinations in Cluster 1', fontsize=10)
	Services_plot.ax[2].tick_params(axis='x', labelsize=8)
	Services_plot.ax[2].set_xlabel('Count', fontsize=8)
	Services_plot.ax[2].set_yticklabels('')
	
	offset = 0
	for x in top_combos.index:
		Services_plot.ax[2].text(15, offset, x, fontsize=8, va='center')
		offset += 1
	
	legend_txt = 'Service Abbreviations\n\n'
	for i, t in enumerate(abrv.items()):
		legend_txt += f"{t[0]}: {t[1]}\n"

	Services_plot.ax[2].text(200, 4, legend_txt, fontsize=8, va='top')
	Services_plot.show()
	Services_plot.save('CA_Contract_Services.png')
ContractServicesAnalysis(df_with_clusters)

## Tenure and Monthly Charge Analysis
def TenureChargesAnalysis(df: pd.DataFrame) -> None:
	"""
    Analyze financial patterns across customer clusters with detailed focus on Cluster 1.
    
    Business Context:
    - Customer lifetime value (CLV) optimization requires understanding charge patterns
    - Extra charges can be indicators of service issues and dissatisfaction
    - Revenue efficiency varies significantly between contract types
    - Price sensitivity differs across customer segments
    - Tenure and spending patterns reveal natural upsell/cross-sell timing
    
    Analysis Goals:
    1. Identify revenue efficiency differences between customer segments
    2. Quantify the financial impact of contract length on customer value
    3. Discover price sensitivity thresholds across different segments
    4. Pinpoint optimal revenue-to-tenure ratios for sustainable customer relationships
    5. Support pricing strategy refinement and financial forecasting
    
    Key Methods:
    - Comparative analysis of charge metrics across clusters
    - Revenue efficiency calculations to identify high-value segments
    - Extra charges analysis as percentage of total revenue
    - Contract impact on financial performance
    - Correlation analysis between tenure and spending patterns
    - Identification of highest-value customer profiles
    """
    
	print(
		"\n"+"#"*20,
		"# CHARGES ANALYSIS #", 
		"#"*20,
		sep="\n", end="\n\n" ,file=OUT
	)
    
	# ===== 1. CHARGES AND TENURE ANALYSIS OVERALL =====
    # 1.1 Charges statistics by cluster
	charges_cols = ['Monthly_Charge', 'Total_Charges', 'Total_Extra_Data_Charges', 
                    'Total_Long_Distance_Charges', 'Total_Revenue']
	charges_summary = df.groupby('Cluster')[charges_cols].agg(['mean', 'median', 'std']).round(2)
	print(f"\n## Charges Overview by Cluster:\n{charges_summary}", file=OUT)
    
    # 1.2 Revenue per tenure month by cluster (customer value efficiency)
	df['Revenue_Per_Month'] = df['Total_Revenue'] / df['Tenure_in_Months']
	revenue_efficiency = df.groupby('Cluster')['Revenue_Per_Month'].agg(['mean', 'median']).round(2)
	print(f"\n## Revenue Efficiency ($ per month):\n{revenue_efficiency}", file=OUT)
    
    # 1.3 Extra charges as percentage of total revenue
	df['Extra_Charges_Pct'] = (df['Total_Extra_Data_Charges'] + df['Total_Long_Distance_Charges']) / df['Total_Revenue'] * 100
	extra_charges_pct = df.groupby('Cluster')['Extra_Charges_Pct'].agg(['mean', 'median']).round(2)
	print(f"\n## Extra Charges as % of Total Revenue:\n{extra_charges_pct}", file=OUT)

	# 1.4 Charges correlation with tenure across clusters
	tenure_corr = df.groupby('Cluster').apply(lambda x: 
        x[['Tenure_in_Months', 'Monthly_Charge', 'Total_Charges']].corr().iloc[0, 1:])
	print(f"\n## Correlation between Tenure and Charges:\n{tenure_corr}", file=OUT)
    

    # ===== 2. CLUSTER 1 DETAILED ANALYSIS =====
	# 2.1 Detailed analysis of Cluster 1
	cluster1 = df[df['Cluster'] == 1]
    
	# 2.2 Charges by contract type in Cluster 1
	contract_charges = cluster1.groupby('Contract')[charges_cols].mean().round(2)
	print(f"\n## Cluster 1: Charges by Contract Type:\n{contract_charges}", file=OUT)
    
    # 2.3 Charges by internet service type in Cluster 1
	internet_charges = cluster1.groupby('Internet_Type')[charges_cols].mean().round(2)
	print(f"\n## Cluster 1: Charges by Internet Service Type:\n{internet_charges}", file=OUT)
    
    # 2.4 Top paying customers in Cluster 1
	top_revenue = cluster1.nlargest(10, 'Total_Revenue')[['Total_Revenue', 'Monthly_Charge', 
                                                         'Tenure_in_Months', 'Contract']]
	print(f"\n## Cluster 1: Top 10 Revenue Customers:\n{top_revenue}", file=OUT)
    

    # ===== 3. VISUALIZATIONS =====
	# 3.1 Monthly charges by cluster
	ChargesPlot = Plot(figsize=(16, 8), ncols=2, title='Charges Analysis')
    
    # 3.2 Box plot of monthly charges by cluster
	sns.boxplot(x='Cluster', y='Monthly_Charge', data=df, ax=ChargesPlot.ax[0])
	ChargesPlot.ax[0].set_title('Monthly Charges Distribution by Cluster', fontsize=12)
	ChargesPlot.ax[0].set_ylabel('Monthly Charges ($)', fontsize=10)
    
    # 3.3 Scatter plot of tenure vs. monthly charges for Cluster 1
	sns.scatterplot(x='Tenure_in_Months', y='Monthly_Charge', 
                    hue='Contract', data=cluster1, ax=ChargesPlot.ax[1])
	ChargesPlot.ax[1].set_title('Cluster 1: Monthly Charges vs. Tenure', fontsize=12)
	ChargesPlot.ax[1].set_xlabel('Tenure (Months)', fontsize=10)
	ChargesPlot.ax[1].set_ylabel('Monthly Charges ($)', fontsize=10)
	ChargesPlot.save('CA_TenureCharges.png')
TenureChargesAnalysis(df_with_clusters)

## Total Charges and Payment Method Analysis
def PaymentMethodAnalysis(df: pd.DataFrame) -> None:
	"""
    Analyze payment methods across clusters with detailed focus on high-risk Cluster 1.
    Examines relationships between payment methods, charges, and customer behavior.

    Business Context:
    - Electronic check payments have shown higher churn rates historically
    - Automatic payment methods correlate with customer loyalty
    - Payment method transitions can indicate churn risk
    - Different payment methods have varying operational costs and reliability
    - Payment friction is a significant contributor to passive churn
    
    Analysis Goals:
    1. Identify high-churn payment channels
    2. Quantify financial impact of each payment method
    3. Discover opportunities for payment method conversion campaigns
    4. Understand payment method preferences across different customer segments
    5. Identify potential payment friction points contributing to churn
    
    Key Methods:
    - Payment method distribution analysis across customer segments
    - Churn rate calculation by payment method for each cluster
    - Financial metrics comparison across different payment types
    - Contract-payment method relationship analysis in high-risk segments
    - Internet service-payment method correlation examination
    - Tenure duration analysis by payment method type
    - Extra charges pattern identification by payment method
    - Customer value and payment method preference correlation
    """
	print(
		"\n"+"#"*27,
		"# PAYMENT METHOD ANALYSIS #", 
		"#"*27,
		sep="\n", end="\n\n", file=OUT
		)
    
	# ===== 1. PAYMENT METHOD ANALYSIS OVERALL =====
    # 1.1 Payment method distribution by cluster
	payment_dist = create_percentage_crosstab(df, 'Cluster', 'Payment_Method')
	print(f"\n## Payment Method Distribution by Cluster (%):\n{payment_dist.round(2)}", file=OUT)
    
    # 1.2 Compare churn rates by payment method across all clusters
	churn_by_payment = df.groupby(['Cluster', 'Payment_Method'])['Customer_Status'].apply(
        lambda x: (x == 'Churned').mean() * 100
    ).unstack().round(2)
    
	print(f"\n## Churn Rate by Payment Method and Cluster (%):\n{churn_by_payment}", file=OUT)


	# ===== 2. CLUSTER 1 DETAILED ANALYSIS =====
	df['Total_Extra_Charges'] = df['Total_Extra_Data_Charges'] + df['Total_Long_Distance_Charges'] 
	cluster1 = df[df['Cluster'] == 1]
    
    # 2.1 Financial metrics by payment method in Cluster 1
	payment_metrics = cluster1.groupby('Payment_Method').agg({
        'Monthly_Charge': 'mean',
        'Total_Charges': 'mean',
		'Total_Extra_Charges': 'mean', 
        'Total_Revenue': 'mean',
        'Tenure_in_Months': 'mean',
        'Customer_Status': lambda x: (x == 'Churned').mean() * 100 
    }).round(2)
    
	payment_metrics.columns = ['Monthly Charge Avg', 'Total Charges Avg', 
                             'Total Extra Charges %', 'Total Revenue Avg', 
                             'Tenure Avg (months)', 'Churn %']
    
	print(f"\n## Cluster 1: Financial Metrics by Payment Method:\n{payment_metrics}", file=OUT)
    
    # 2.2 Payment methods by contract type in Cluster 1
	contract_payment = create_percentage_crosstab(cluster1, 'Contract', 'Payment_Method') 
	print(f"\n## Cluster 1: Payment Method by Contract Type (%):\n{contract_payment.round(2)}", file=OUT)
    
    # 2.3 Payment methods by internet service type in Cluster 1
	internet_payment = create_percentage_crosstab(cluster1, 'Internet_Type', 'Payment_Method')   
	print(f"\n## Cluster 1: Payment Method by Internet Type (%):\n{internet_payment.round(2)}", file=OUT)
    
    
	# ===== 3. VISUALIZATIONS =====
	PaymentPlot = Plot(ncols=2, nrows=2, title='Payment Methods')
    
    # 3.1. Payment method distribution by cluster
	payment_dist.plot(kind='bar', stacked=True, ax=PaymentPlot.ax[0], rot=0)
	PaymentPlot.ax[0].set_title('Payment Method Distribution by Cluster', fontsize=12)
	PaymentPlot.ax[0].set_ylabel('Percentage (%)', fontsize=10)
	PaymentPlot.ax[0].legend(title='Payment Method')
    
    # 3.2. Churn rate by payment method in Cluster 1
	churned_c1 = cluster1[cluster1['Customer_Status'] == 'Churned']
	churn_by_method = churned_c1['Payment_Method'].value_counts() * 100
	churn_by_method.plot(kind='bar', ax=PaymentPlot.ax[1], rot=45)    
	PaymentPlot.ax[1].set_title('Cluster 1: Churn Rate by Payment Method', fontsize=12)
	PaymentPlot.ax[1].set_ylabel('Churn Rate (%)', fontsize=10)
	    
    # 3.3. Boxplot of total charges by payment method in Cluster 1
	sns.boxplot(x='Payment_Method', y='Total_Charges', data=cluster1, ax=PaymentPlot.ax[2])
	PaymentPlot.ax[2].set_title('Cluster 1: Total Charges by Payment Method', fontsize=12)
	PaymentPlot.ax[2].set_ylabel('Total Charges ($)', fontsize=10)
    
    # 3.4. Tenure by payment method in Cluster 1
	sns.boxplot(x='Payment_Method', y='Tenure_in_Months', data=cluster1, ax=PaymentPlot.ax[3])
	PaymentPlot.ax[3].set_title('Cluster 1: Tenure by Payment Method', fontsize=12)
	PaymentPlot.ax[3].set_ylabel('Tenure (Months)', fontsize=10)
	PaymentPlot.save('CA_Payment_Methods.png')
PaymentMethodAnalysis(df_with_clusters)

## Customer satisfaction Analysis
def SatisfactionAnalysis(df: pd.DataFrame) -> None:
    """
    Analyze customer satisfaction across clusters with detailed focus on high-risk Cluster 1.
    
    Business Context:
    - Satisfaction scores are leading indicators of future churn risk
    - Service quality perceptions vary significantly across customer segments
    - Contract structure impacts satisfaction differently across segments
    - Payment methods correlate with satisfaction due to friction/convenience factors
    - Understanding satisfaction drivers enables proactive retention initiatives
    
    Analysis Goals:
    1. Quantify the relationship between satisfaction scores and churn probability
    2. Identify which services have strongest impact on satisfaction in high-risk segments
    3. Discover demographic patterns in satisfaction ratings
    4. Determine how contract types influence satisfaction perception
    5. Support customer experience improvement initiatives and retention program design
    
    Key Methods:
    - Satisfaction distribution analysis across customer segments
    - Correlation between satisfaction scores and churn behavior
    - Demographic impact on satisfaction ratings
    - Service adoption patterns and their effect on satisfaction
    - Contract type influence on customer satisfaction
    - Payment method correlation with satisfaction levels
    """

	# ===== 1. SATISFACTION ANALYSIS OVERALL =====
    print(
		"\n"+"#"*34,
		"# CUSTOMER SATISFACTION ANALYSIS #",
		"#"*34,
		sep="\n", end="\n\n", file=OUT)
    
    # 1.1 Basic satisfaction statistics by cluster
    satisfaction_summary = df.groupby('Cluster')['Satisfaction_Score'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(f"\n## Satisfaction Overview by Cluster:\n{satisfaction_summary}", file=OUT)
    
    # 1.2 Satisfaction distribution by cluster (percentage in each score category)
    sat_dist = create_percentage_crosstab(df, 'Cluster', 'Satisfaction_Score')
    print(f"\n## Satisfaction Distribution by Cluster (%):\n{sat_dist.round(2)}", file=OUT)
    
    # 1.3 Relationship between satisfaction and churn
    sat_churn = df.groupby(['Satisfaction_Score', 'Customer_Status']).size().unstack()
    sat_churn_pct = sat_churn.div(sat_churn.sum(axis=1), axis=0) * 100
    print(f"\n## Churn Rate by Satisfaction Score (%):\n{sat_churn_pct['Churned'].round(2)}", file=OUT)
    
	# ===== 2. CLUSTER 1 DETAILED ANALYSIS =====
    # 2.1 Detailed analysis of Cluster 1
    cluster1 = df[df['Cluster'] == 1]
    
    # 2.2 Demographics by satisfaction in Cluster 1
    demo_sat = cluster1.groupby('Satisfaction_Score').agg({
        'Gender': 'mean',
        'Age': 'mean',
        'Married': [lambda x: (x==1).mean() * 100],
        'Dependents': [lambda x: (x==1).mean() * 100],
        'Tenure_in_Months': 'mean'
    }).round(2)
    
    print(f"\n## Cluster 1: Demographics by Satisfaction Score:\n{demo_sat}", file=OUT)
    
    # 2.3 Services impact on satisfaction in Cluster 1
    service_cols = [col for col in cluster1.columns if any(
        service in col.lower() for service in ['phone', 'internet', 'security', 'backup', 
                                               'protection', 'tech', 'stream', 'unlimited']
    )]
    binary_service_cols = [col for col in service_cols if cluster1[col].dtype != 'object']
    if binary_service_cols:
        services_sat = cluster1.groupby('Satisfaction_Score')[binary_service_cols].mean() * 100
        print(f"\n## Cluster 1: Services by Satisfaction Score (%):\n{services_sat.round(2)}", file=OUT)
    
    # 2.4 Contract type impact on satisfaction in Cluster 1
    contract_sat = create_percentage_crosstab(cluster1, 'Contract', 'Satisfaction_Score')
    print(f"\n## Cluster 1: Satisfaction by Contract Type (%):\n{contract_sat.round(2)}", file=OUT)
    
    # 2.5 Payment method impact on satisfaction in Cluster 1
    payment_sat = create_percentage_crosstab(cluster1, 'Payment_Method', 'Satisfaction_Score')    
    print(f"\n## Cluster 1: Satisfaction by Payment Method (%):\n{payment_sat.round(2)}", file=OUT)
    
    # 2.6 Charges by satisfaction in Cluster 1
    charges_sat = cluster1.groupby('Satisfaction_Score').agg({
        'Monthly_Charge': 'mean',
        'Total_Charges': 'mean',
        'Total_Revenue': 'mean',
        'Total_Extra_Data_Charges': 'mean',
        'Total_Long_Distance_Charges': 'mean'
    }).round(2)
    
    print(f"\n## Cluster 1: Charges by Satisfaction Score:\n{charges_sat}", file=OUT)
    
    # 2.7 Correlations between satisfaction and other metrics
    corr_metrics = ['Satisfaction_Score', 'Churn_Score', 'CLTV', 
                   'Tenure_in_Months', 'Monthly_Charge', 'Total_Revenue']
    corr_by_cluster = df.groupby('Cluster')[corr_metrics].corr().loc[pd.IndexSlice[:, 'Satisfaction_Score'], :]
    corr_by_cluster = corr_by_cluster.drop('Satisfaction_Score', axis=1).reset_index(level=1, drop=True)
    
    print(f"\n## Correlation between Satisfaction and Key Metrics by Cluster:\n{corr_by_cluster.round(3)}", file=OUT)
    

    # ===== 3. VISUALIZATIONS =====
    SatPlot = Plot(figsize=(16, 10), ncols=2, nrows=2, title='Customer Satisfaction')
    # 3.1. Satisfaction distribution by cluster
    sat_dist.plot(kind='bar', stacked=True, ax=SatPlot.ax[0], rot=0)
    SatPlot.ax[0].set_title('Satisfaction Score Distribution by Cluster', fontsize=12)
    SatPlot.ax[0].set_ylabel('Percentage (%)', fontsize=10)
    SatPlot.ax[0].set_xlabel('Cluster', fontsize=10)
    SatPlot.ax[0].legend(title='Satisfaction')
    
    # 3.2. Churn rate by satisfaction score
    sat_churn_pct['Churned'].plot(kind='bar', ax=SatPlot.ax[1], rot=0, color='crimson')
    SatPlot.ax[1].set_title('Churn Rate by Satisfaction Score', fontsize=12)
    SatPlot.ax[1].set_ylabel('Churn Rate (%)', fontsize=10)
    SatPlot.ax[1].set_xlabel('Satisfaction Score', fontsize=10)
    
    # 3.3. Heatmap of satisfaction vs contract in Cluster 1
    sns.heatmap(contract_sat, annot=True, cmap='YlGnBu', fmt='.1f', ax=SatPlot.ax[2])
    SatPlot.ax[2].set_title('Cluster 1: Satisfaction by Contract Type (%)', fontsize=12)
    
    # 3.4. Average satisfaction by tenure groups
    tenure_bins = [0, 12, 24, 36, 48, 60, 72]
    tenure_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    cluster1['Tenure_Group'] = pd.cut(cluster1['Tenure_in_Months'], bins=tenure_bins, labels=tenure_labels)
    tenure_sat = cluster1.groupby('Tenure_Group')['Satisfaction_Score'].mean().reset_index()
    sns.barplot(x='Tenure_Group', y='Satisfaction_Score', data=tenure_sat, ax=SatPlot.ax[3])
    SatPlot.ax[3].set_title('Cluster 1: Avg Satisfaction by Tenure (months)', fontsize=12)
    SatPlot.ax[3].set_ylim(1, 5)
    SatPlot.ax[3].set_ylabel('Avg Satisfaction Score', fontsize=10)
    SatPlot.ax[3].set_xlabel('Tenure Group (months)', fontsize=10)
    
    SatPlot.save('CA_Satisfaction.png')
SatisfactionAnalysis(df_with_clusters)






''' 6. DATA STORAGE '''

# Close the output file
print(f"Report saved to {reports_dir}\CohortAnalysis.txt")
OUT.close()
OUT = None

# Save the final dataset to a CSV file for further analysis and modeling
df_original_copy[binary_features] = df[binary_features]
df_transformed = pd.concat([df_original_copy, df_with_clusters['Cluster']], axis=1)
filename:Path = CONFIG['input_dir'] / 'telco_customers_transformed.csv'
print(f"\nSaving the transformed dataset to {filename}", end='\n\n')
df_transformed.to_csv(filename, index=False)

# Open a connection to the database
connector	= SQLServerConnector(CONFIG['db_config']['db_name'])
conn 		= connector.connect()
cursor 		= conn.cursor()

# Create table CustomerClusters in the database
try:
	query1	= """
			USE CustomerChurnDB;
			IF OBJECT_ID('[CustomerChurnDB].[TelcoChurnQ3].[Clusters]') IS NOT NULL 
				DROP TABLE TelcoChurnQ3.Clusters;
			CREATE TABLE TelcoChurnQ3.Clusters(
				ClusterID INT PRIMARY KEY,
				Description VARCHAR(255)
			);
			"""
	print(f"Creating table 'TelcoChurnQ3.Clusters' in the database", end='\n\n')
	connector.push_data(query1)
except Exception as e:
	print(f"Error: {e}", end='\n\n')

# Merge cluster IDs into the table
try:
    for cluster_id in range(df_transformed['Cluster'].nunique()):
        cursor.execute("""
            MERGE TelcoChurnQ3.Clusters AS target
            USING (SELECT ? AS ClusterID, ? AS Description) AS source
            ON target.ClusterID = source.ClusterID
            WHEN MATCHED THEN
                UPDATE SET target.Description = source.Description
            WHEN NOT MATCHED THEN
                INSERT (ClusterID, Description) 
                VALUES (source.ClusterID, source.Description);
        """, [cluster_id, f"Cluster {cluster_id}"])
    conn.commit()
    print(f"Merged {df_transformed['Cluster'].nunique()} cluster IDs into 'TelcoChurnQ3.Clusters'", end='\n\n')
except Exception as e:
    conn.rollback()
    print(f"Error: {e}", end='\n\n')

# Add ClusterID column to the CustomerInfo table
try:
	query2 	= """
			IF NOT EXISTS (
				SELECT 1 FROM sys.columns 
				WHERE object_id = OBJECT_ID('TelcoChurnQ3.CustomerInfo') 
				AND name = 'ClusterID'
			) ALTER TABLE TelcoChurnQ3.CustomerInfo ADD ClusterID INT;
			"""
	query3 	= """
			IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_ClusterID')
				ALTER TABLE TelcoChurnQ3.CustomerInfo 
				ADD CONSTRAINT FK_ClusterID FOREIGN KEY (ClusterID)
				REFERENCES TelcoChurnQ3.Clusters(ClusterID)
				ON DELETE CASCADE ON UPDATE CASCADE;
			"""
	connector.push_data(query2)
	connector.push_data(query3)
	print(f"Added 'ClusterID' column to 'TelcoChurnQ3.CustomerInfo'", end='\n\n')
except Exception as e:
	print(f"Error: {e}", end='\n\n')

# Insert cluster values into tables
clust_rcrds	= pd.concat([ df_transformed['Cluster'], df_original_copy['Customer_ID'] ] , axis=1).values.tolist()
try:
	query = "UPDATE TelcoChurnQ3.CustomerInfo SET ClusterID = ? WHERE CustomerID = ?"
	cursor.executemany(query, clust_rcrds)
	conn.commit()
	print(f"Updated {len(clust_rcrds)} records into 'TelcoChurnQ3.CustomerInfo'", end='\n\n')
except Exception as e:
	print(f"Error: {e}", end='\n\n')

# Close the connection
cursor.close()
connector.close()
cursor = None
connector = None

