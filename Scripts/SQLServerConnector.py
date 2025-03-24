#!/usr/bin env python
# -*- coding: utf-8 -*-

"""
Author: Michael Garancher
Date: 2025-03-01
Description: Connects to a local SQL Server database using a trusted connection.
"""

import sys
import os
import warnings
import pyodbc
import pandas as pd

sys.tracebacklimit = 0	


class SQLServerConnector:
	""" 
	Args:
		database (str): The name of the database to connect to.
	Attributes:
		driver (str): The ODBC driver to use for the connection.
		server (str): The name of the server to connect to.
		database (str): The name of the database to connect to.
		conn (pyodbc.Connection): A connection object to the SQL Server database.
	Methods:
		connect: Connect to the SQL Server database.
		close: Close the connection to the SQL Server database.
		fetch_data: Fetch data from the SQL Server database into a DataFrame object.
		push_data: Push data to the SQL Server database.
	Returns:
		pyodbc.Connection: A connection object to the SQL Server database.
	"""
	_instance = None
	def __init__(self, database:str) -> object:
		self._driver = '{ODBC Driver 17 for SQL Server}'
		self._server = os.environ.get('COMPUTERNAME')
		self.database = database
		self.conn = None
	
	def __new__(cls, database:str=None) -> object:
		if not database:
			raise ValueError("Database cannot be empty")
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	@property
	def driver(self) -> str:
		return self._driver

	@driver.setter
	def driver(self, driver:str) -> None:
		if not driver:
			raise ValueError("Driver cannot be empty")
		self._driver = driver

	@property
	def server(self) -> str:
		return self._server

	@server.setter
	def server(self, server:str) -> None:
		if not server:
			raise ValueError("Server cannot be empty")
		self._driver = server

	def connect(self) -> pyodbc.Connection:
		try:
			self.conn = pyodbc.connect(
				f'DRIVER={self.driver};'
				f'SERVER={self.server};'
				f'DATABASE={self.database};'
				'Trusted_Connection=yes;'
			)
		except Exception as e:
			raise Exception(e)
		else:
			print(f"\nSuccessfully connected to database {self.database}")
			return self.conn
	
	def close(self) -> None:
		if self.conn:
			self.conn.close()
			print(f"Connection to {self.database} closed", end="\n\n")
			
	def fetch_data(self, query: str) -> pd.DataFrame:
		if self.conn is None:
			raise ConnectionError("Not connected to any database")
		else:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				df = pd.read_sql(query, self.conn)
				print("Data fetched successfully")
				return df
	def push_data(self, query:str) -> None:
		if self.conn is None:
			raise ConnectionError("Not connected to any database")
		else:
			self.conn.execute(query)
			self.conn.commit()
			print("Query executed successfully")	