/* SQL script to achieve 3NF for the Customer Churn project */
-- NB: The Quarter attribute has been eliminated since the whole dataset is derived from the same fiscal quarter (Q3)


-- Go to the CustomerChurnDB database
USE CustomerChurnDB;
GO

-- Create a dedicated schema
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'TelcoChurnQ3')
BEGIN
    EXEC('CREATE SCHEMA TelcoChurnQ3')
END
GO

-- Drop tables wrt foreign key constraints
IF OBJECT_ID('TelcoChurnQ3.CustomerChurnHistory') IS NOT NULL
    DROP TABLE TelcoChurnQ3.CustomerChurnHistory;

IF OBJECT_ID('TelcoChurnQ3.CustomerBilling') IS NOT NULL
    DROP TABLE TelcoChurnQ3.CustomerBilling;
    
IF OBJECT_ID('TelcoChurnQ3.CustomerContract') IS NOT NULL
    DROP TABLE TelcoChurnQ3.CustomerContract;
    
IF OBJECT_ID('TelcoChurnQ3.CustomerServices') IS NOT NULL
    DROP TABLE TelcoChurnQ3.CustomerServices;
    
IF OBJECT_ID('TelcoChurnQ3.CustomerInfo') IS NOT NULL
    DROP TABLE TelcoChurnQ3.CustomerInfo;

IF OBJECT_ID('TelcoChurnQ3.ChurnReasons') IS NOT NULL
    DROP TABLE TelcoChurnQ3.ChurnReasons;
    
IF OBJECT_ID('TelcoChurnQ3.PaymentMethods') IS NOT NULL
    DROP TABLE TelcoChurnQ3.PaymentMethods;
    
IF OBJECT_ID('TelcoChurnQ3.ContractTypes') IS NOT NULL
    DROP TABLE TelcoChurnQ3.ContractTypes;
    
IF OBJECT_ID('TelcoChurnQ3.InternetServices') IS NOT NULL
    DROP TABLE TelcoChurnQ3.InternetServices;
    
IF OBJECT_ID('TelcoChurnQ3.Locations') IS NOT NULL
    DROP TABLE TelcoChurnQ3.Locations;


-- Create Locations Table
IF OBJECT_ID('TelcoChurnQ3.Locations') IS NOT NULL
	DROP TABLE TelcoChurnQ3.Locations;
CREATE TABLE TelcoChurnQ3.Locations (
    LocationID INT IDENTITY(1,1) PRIMARY KEY,
    Country VARCHAR(50),
    State VARCHAR(50),
    City VARCHAR(50),
    ZipCode VARCHAR(20),
    Population INT NULL,
    Latitude DECIMAL(9,6) NULL,
    Longitude DECIMAL(9,6) NULL
);
GO

-- Create Internet Services Table
IF OBJECT_ID('TelcoChurnQ3.InternetServices') IS NOT NULL
	DROP TABLE TelcoChurnQ3.InternetServices;
CREATE TABLE TelcoChurnQ3.InternetServices (
    InternetServiceID INT IDENTITY(1,1) PRIMARY KEY,
    InternetServiceType VARCHAR(50) UNIQUE -- {DSL, Fiber Optic, No Internet | None}
);
GO

-- Create Contract Types Table
IF OBJECT_ID('TelcoChurnQ3.ContractTypes') IS NOT NULL
	DROP TABLE TelcoChurnQ3.ContractTypes;
CREATE TABLE TelcoChurnQ3.ContractTypes (
    ContractTypeID INT IDENTITY(1,1) PRIMARY KEY,
    ContractTypeName VARCHAR(50) UNIQUE -- {Month-to-Month, One Year, Two Year}
);
GO

-- Create Payment Methods Table
IF OBJECT_ID('TelcoChurnQ3.PaymentMethods') IS NOT NULL
	DROP TABLE TelcoChurnQ3.PaymentMethods;
CREATE TABLE TelcoChurnQ3.PaymentMethods (
    PaymentMethodID INT IDENTITY(1,1) PRIMARY KEY,
    PaymentMethodName VARCHAR(50) UNIQUE -- {Credit Card, Bank Widthdrawal, Mailed Check}
);
GO

-- Create Churn Reasons Table
IF OBJECT_ID('TelcoChurnQ3.ChurnReasons') IS NOT NULL
	DROP TABLE TelcoChurnQ3.ChurnReasons;
CREATE TABLE TelcoChurnQ3.ChurnReasons (
    ChurnReasonID INT IDENTITY(1,1) PRIMARY KEY,
    ChurnReason VARCHAR(255) UNIQUE
);

-- Create CustomersInfo Table
IF OBJECT_ID('TelcoChurnQ3.CustomerInfo') IS NOT NULL
	DROP TABLE TelcoChurnQ3.CustomersInfo;
CREATE TABLE TelcoChurnQ3.CustomerInfo (
    CustomerID NVARCHAR(50) NOT NULL PRIMARY KEY,
    Gender BIT, -- {1: 'Male', 0:'Female'}
    Age INT,
    SeniorCitizen BIT,
    UnderThirty BIT,
    Married BIT,
	Dependents BIT,
	NumberOfDependents INT DEFAULT 0,
	ReferredAFriend BIT,
	NumberOfReferrals INT DEFAULT 0,
	LocationID INT,
    Status VARCHAR(50) CHECK (Status IN ('Churned', 'Stayed', 'Joined')),
    SatisfactionScore INT CHECK (SatisfactionScore BETWEEN 1 AND 5),
	CLTV INT,
	CONSTRAINT FK_LocationID FOREIGN KEY (LocationID) 
		REFERENCES TelcoChurnQ3.Locations(LocationID)
		ON DELETE CASCADE ON UPDATE CASCADE
);
GO

-- Create Customer Subscribed Services Table 
IF OBJECT_ID('TelcoChurnQ3.CustomerServices') IS NOT NULL
	DROP TABLE TelcoChurnQ3.CustomerServices;
CREATE TABLE TelcoChurnQ3.CustomerServices (
    CustomerServiceID INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID NVARCHAR(50) NOT NULL,
    PhoneService BIT,
    MultipleLines BIT,
    InternetServiceID INT NULL,
    OnlineSecurity BIT,
    OnlineBackup BIT,
    DeviceProtection BIT,
    PremiumTechSupport BIT,
    StreamingTV BIT,
    StreamingMovies BIT,
	StreamingMusic BIT,
	UnlimitedData BIT,
	CONSTRAINT FK_CustServ_Customers FOREIGN KEY (CustomerID)
       REFERENCES TelcoChurnQ3.CustomerInfo(CustomerID)
	   ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT FK_CustServ_InternetServices FOREIGN KEY (InternetServiceID)
       REFERENCES TelcoChurnQ3.InternetServices(InternetServiceID)
	   ON DELETE CASCADE ON UPDATE CASCADE
);
GO

-- Create Customer Contracts Table (Contract & Engagement Details)
IF OBJECT_ID('TelcoChurnQ3.CustomerContract') IS NOT NULL
	DROP TABLE TelcoChurnQ3.CustomerContracts;
CREATE TABLE TelcoChurnQ3.CustomerContract (
    CustomerContractID INT IDENTITY(1,1) PRIMARY KEY,
	CustomerID NVARCHAR(50), 
    ContractTypeID INT,
	OfferType NVARCHAR(50), -- {A, B, C, D, E, None}
    PaperlessBilling BIT,
    PaymentMethodID INT,	
	TenureInMonths INT,
	CONSTRAINT FK_Contracts_Customers FOREIGN KEY (CustomerID)
	   REFERENCES TelcoChurnQ3.CustomerInfo(CustomerID)
	   ON DELETE CASCADE ON UPDATE CASCADE,
	CONSTRAINT FK_Contracts_ContractTypes FOREIGN KEY (ContractTypeID)
	   REFERENCES TelcoChurnQ3.ContractTypes(ContractTypeID)
	   ON DELETE CASCADE ON UPDATE CASCADE,
	CONSTRAINT FK_Contracts_PaymentMethods FOREIGN KEY (PaymentMethodID)
	   REFERENCES TelcoChurnQ3.PaymentMethods(PaymentMethodID)
	   ON DELETE CASCADE ON UPDATE CASCADE
);
GO

-- Create Customer Billing Table
IF OBJECT_ID('TelcoChurnQ3.CustomerBilling') IS NOT NULL
	DROP TABLE TelcoChurnQ3.CustomerBilling;
CREATE TABLE TelcoChurnQ3.CustomerBilling (
    BillingID INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID NVARCHAR(50) NOT NULL,
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(10,2), -- MonthlyCharge * TenureInMonths ? Values from the dataset are not consistent with this formula
    TotalExtraDataCharges DECIMAL(10,2) DEFAULT 0,
	TotalLongDistanceCharges DECIMAL(10,2) DEFAULT 0,
	TotalRevenue DECIMAL(10,2), -- TotalCharges + TotalExtraCharges
	CONSTRAINT FK_Billing_Customers FOREIGN KEY (CustomerID)
       REFERENCES TelcoChurnQ3.CustomerInfo(CustomerID)
	   ON DELETE CASCADE ON UPDATE CASCADE
);
GO

-- Create Customer Churn History Table
IF OBJECT_ID('TelcoChurnQ3.CustomerChurnHistory') IS NOT NULL
	DROP TABLE TelcoChurnQ3.CustomerChurnHistory;
CREATE TABLE TelcoChurnQ3.CustomerChurnHistory (
    ChurnID INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID NVARCHAR(50) NOT NULL,
    ChurnLabel BIT,
	ChurnCategory NVARCHAR(50), -- {Attitude, Competitor, Dissatisfaction, Other, Price, NULL}
    ChurnReasonID INT,
	ChurnScore INT CHECK (ChurnScore BETWEEN 0 AND 100),
	CONSTRAINT FK_ChurnHist_Customers FOREIGN KEY (CustomerID)
       REFERENCES TelcoChurnQ3.CustomerInfo(CustomerID),
    CONSTRAINT FK_ChurnHist_ChurnReasons FOREIGN KEY (ChurnReasonID)
       REFERENCES TelcoChurnQ3.ChurnReasons(ChurnReasonID)
);
GO