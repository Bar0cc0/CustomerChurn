/* Import the transformed dataset from a CSV file into the TelcoChurnQ3 schema */


SET NOCOUNT ON; 
GO

-- Enable xp_cmdshell and required configuration
IF NOT EXISTS (SELECT * FROM sys.configurations WHERE name = 'xp_cmdshell' AND value_in_use = 1)
BEGIN
    EXEC sp_configure 'show advanced options', 1;  
    RECONFIGURE;
    EXEC sp_configure 'xp_cmdshell', 1;
    RECONFIGURE;
END;
GO

-- Set context
USE CustomerChurnDB;
GO

-- Create TelcoChurnQ3 schema if it doesn't exist
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'TelcoChurnQ3')
BEGIN
    EXEC('CREATE SCHEMA TelcoChurnQ3');
    PRINT 'Schema TelcoChurnQ3 created successfully.';
END
GO

-- Create error log table 
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Log' AND schema_id = SCHEMA_ID('TelcoChurnQ3'))
BEGIN
    CREATE TABLE TelcoChurnQ3.Log (
        ErrorLogID INT IDENTITY(1,1) PRIMARY KEY,
        ErrorTime DATETIME DEFAULT GETDATE(),
        ErrorNumber INT,
        ErrorSeverity INT,
        ErrorState INT,
        ErrorProcedure NVARCHAR(128),
        ErrorLine INT,
        ErrorMessage NVARCHAR(4000),
        Operation NVARCHAR(200)
    );
END
GO

-- Declare local variables
DECLARE @ROOTDIR NVARCHAR(255) = 'D:\Projects\PortefolioWebsite\Projects\CustomerChurn\Datasets\' --'$(Param)\Datasets\';
DECLARE @FilePath NVARCHAR(512) = @ROOTDIR + 'telco_customers_transformed.csv';
DECLARE @SQL NVARCHAR(MAX);
DECLARE @ErrorMessage NVARCHAR(4000);
DECLARE @ErrorSeverity INT;
DECLARE @ErrorState INT;
DECLARE @FileExists INT;

-- Check if the filepath exists
EXEC master.dbo.xp_fileexist @FilePath, @FileExists OUTPUT;
IF ISNULL(@FileExists, 0) = 0
BEGIN
    RAISERROR('File %s does not exist', 16, 1, @FilePath);
    RETURN;
END

-- Create a temporary table to hold all CSV data
-- Avoid syntax error caused by the WITH clause in OPENROWSET statements with SQLServer 2017 < 14.0.3048

-- 
IF OBJECT_ID('tempdb..#TelcoData') IS NOT NULL
    DROP TABLE #TelcoData;

CREATE TABLE #TelcoData (
    col1 NVARCHAR(255), col2 NVARCHAR(255), col3 NVARCHAR(255), 
    col4 NVARCHAR(255), col5 NVARCHAR(255), col6 NVARCHAR(255), 
    col7 NVARCHAR(255), col8 NVARCHAR(255), col9 NVARCHAR(255), 
    col10 NVARCHAR(255), col11 NVARCHAR(255), col12 NVARCHAR(255), 
    col13 NVARCHAR(255), col14 NVARCHAR(255), col15 NVARCHAR(255), 
    col16 NVARCHAR(255), col17 NVARCHAR(255), col18 NVARCHAR(255), 
    col19 NVARCHAR(255), col20 NVARCHAR(255), col21 NVARCHAR(255), 
    col22 NVARCHAR(255), col23 NVARCHAR(255), col24 NVARCHAR(255), 
    col25 NVARCHAR(255), col26 NVARCHAR(255), col27 NVARCHAR(255), 
    col28 NVARCHAR(255), col29 NVARCHAR(255), col30 NVARCHAR(255), 
    col31 NVARCHAR(255), col32 NVARCHAR(255), col33 NVARCHAR(255), 
    col34 NVARCHAR(255), col35 NVARCHAR(255), col36 NVARCHAR(255), 
    col37 NVARCHAR(255), col38 NVARCHAR(255), col39 NVARCHAR(255), 
    col40 NVARCHAR(255), col41 NVARCHAR(255), col42 NVARCHAR(255), 
    col43 NVARCHAR(255), col44 NVARCHAR(255), col45 NVARCHAR(255), 
    col46 NVARCHAR(255), col47 NVARCHAR(255), col48 NVARCHAR(255), 
    col49 NVARCHAR(255), col50 NVARCHAR(255)
);

-- Bulk insert data into temporary table
BEGIN TRY
    SET @SQL = N'
    BULK INSERT #TelcoData
    FROM ''' + @FilePath + '''
    WITH (
        FIRSTROW = 2,
        FIELDTERMINATOR = '','',
        ROWTERMINATOR = ''\n'',
        TABLOCK
    )';
    
    EXEC sp_executesql @SQL;
    PRINT 'CSV data successfully imported into temporary table.';
END TRY
BEGIN CATCH
    SELECT 
        @ErrorMessage = ERROR_MESSAGE(),
        @ErrorSeverity = ERROR_SEVERITY(),
        @ErrorState = ERROR_STATE();
        
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
    RETURN;
END CATCH;

-- Import the data
/* ROUTINE: Import lookup tables */
BEGIN TRY
    BEGIN TRANSACTION;
    
    -- Update independent tables - Locations
    MERGE INTO TelcoChurnQ3.Locations AS target
    USING (
        SELECT DISTINCT col9 AS Country, col10 AS State, col11 AS City, 
                        col12 AS ZipCode, col15 AS Population, 
                        col13 AS Latitude, col14 AS Longitude
        FROM #TelcoData
    ) AS source ON target.Country = source.Country AND target.State = source.State 
                AND target.City = source.City AND target.ZipCode = source.ZipCode
    WHEN NOT MATCHED THEN 
        INSERT (Country, State, City, ZipCode, Population, Latitude, Longitude)
        VALUES (source.Country, source.State, source.City, source.ZipCode, 
                source.Population, source.Latitude, source.Longitude);
                
    -- PaymentMethods
    MERGE INTO TelcoChurnQ3.PaymentMethods AS target
    USING (
        SELECT DISTINCT col37 AS PaymentMethodName
        FROM #TelcoData
    ) AS source ON target.PaymentMethodName = source.PaymentMethodName
    WHEN NOT MATCHED THEN 
        INSERT (PaymentMethodName) 
        VALUES (source.PaymentMethodName);
        
    -- ContractTypes
    MERGE INTO TelcoChurnQ3.ContractTypes AS target 
    USING (
        SELECT DISTINCT col20 AS ContractTypeName
        FROM #TelcoData
    ) AS source ON target.ContractTypeName = source.ContractTypeName
    WHEN NOT MATCHED THEN
        INSERT (ContractTypeName) 
        VALUES (source.ContractTypeName);
        
    -- ChurnReasons
    MERGE INTO TelcoChurnQ3.ChurnReasons AS target 
    USING ( 
        SELECT DISTINCT col49 AS ChurnReason
        FROM #TelcoData
        WHERE col49 IS NOT NULL AND col49 <> ''
    ) AS source ON target.ChurnReason = source.ChurnReason
    WHEN NOT MATCHED THEN
        INSERT (ChurnReason) 
        VALUES (source.ChurnReason);
        
    -- InternetServices
    MERGE INTO TelcoChurnQ3.InternetServices AS target
    USING (
        SELECT DISTINCT col25 AS InternetServiceType
        FROM #TelcoData
    ) AS source ON target.InternetServiceType = source.InternetServiceType
    WHEN NOT MATCHED THEN
        INSERT (InternetServiceType) 
        VALUES (source.InternetServiceType);

    COMMIT TRANSACTION;
    PRINT 'Lookup tables updated successfully.';
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;

    SELECT 
        @ErrorMessage = ERROR_MESSAGE(),
        @ErrorSeverity = ERROR_SEVERITY(),
        @ErrorState = ERROR_STATE();

    INSERT INTO TelcoChurnQ3.Log (
        ErrorNumber, ErrorSeverity, ErrorState, 
        ErrorProcedure, ErrorLine, ErrorMessage, Operation
    )
    VALUES (
        ERROR_NUMBER(), ERROR_SEVERITY(), ERROR_STATE(), 
        ERROR_PROCEDURE(), ERROR_LINE(), ERROR_MESSAGE(), 'Lookup Tables Import'
    );
        
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
END CATCH;

/* ROUTINE: Import customer data */
BEGIN TRY
    BEGIN TRANSACTION;
    
    -- CustomerInfo
    INSERT INTO TelcoChurnQ3.CustomerInfo (
        CustomerID, Gender, Age, UnderThirty, SeniorCitizen, Married, 
        Dependents, NumberOfDependents, ReferredAFriend, NumberOfReferrals, 
        Status, SatisfactionScore, CLTV, LocationID, ClusterID
    )
    SELECT 
        t.col1, t.col2, t.col3, t.col4, t.col5, 
        t.col6, t.col7, t.col8, t.col17, t.col18, 
        t.col44, t.col43, t.col47, l.LocationID, c.ClusterID
    FROM #TelcoData t
    JOIN TelcoChurnQ3.Locations l
    ON l.Country = t.col9 AND l.State = t.col10 AND l.City = t.col11 AND l.ZipCode = t.col12
	JOIN TelcoChurnQ3.Clusters c ON c.ClusterID = t.col50;

    -- CustomerServices
    INSERT INTO TelcoChurnQ3.CustomerServices (
        CustomerID, PhoneService, MultipleLines, InternetServiceID, 
        OnlineSecurity, OnlineBackup, DeviceProtection, PremiumTechSupport, 
        StreamingTV, StreamingMovies, StreamingMusic, UnlimitedData)
    SELECT 
        t.col1, t.col21, t.col23, i.InternetServiceID,
        t.col27, t.col28, t.col29, t.col30,
        t.col31, t.col32, t.col33, t.col34
    FROM #TelcoData t
    JOIN TelcoChurnQ3.InternetServices i 
    ON i.InternetServiceType = t.col25;

    -- CustomerChurnHistory
    INSERT INTO TelcoChurnQ3.CustomerChurnHistory (
        CustomerID, ChurnLabel, ChurnCategory, ChurnReasonID, ChurnScore
    )
    SELECT 
        t.col1, t.col45, t.col48, r.ChurnReasonID, t.col46
    FROM #TelcoData t
    LEFT JOIN TelcoChurnQ3.ChurnReasons r 
    ON r.ChurnReason = t.col49;

    -- CustomerBilling
    INSERT INTO TelcoChurnQ3.CustomerBilling (
        CustomerID, MonthlyCharges, TotalCharges, TotalExtraDataCharges, 
        TotalLongDistanceCharges, TotalRevenue
    )    
    SELECT 
        t.col1, t.col38, t.col39, t.col40, t.col41, t.col42
    FROM #TelcoData t;

    -- CustomerContracts
    INSERT INTO TelcoChurnQ3.CustomerContract (
        CustomerID, ContractTypeID, OfferType, 
        PaperlessBilling, PaymentMethodID, TenureInMonths)
    SELECT 
        t.col1, c.ContractTypeID, t.col20,
        t.col36, p.PaymentMethodID, t.col19
    FROM #TelcoData t
    JOIN TelcoChurnQ3.ContractTypes c ON c.ContractTypeName = t.col20
    JOIN TelcoChurnQ3.PaymentMethods p ON p.PaymentMethodName = t.col37;

    COMMIT TRANSACTION;
    PRINT 'Dependent tables updated successfully.';
    PRINT 'Data import completed successfully.';
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;
    
    SELECT 
        @ErrorMessage = ERROR_MESSAGE(),
        @ErrorSeverity = ERROR_SEVERITY(),
        @ErrorState = ERROR_STATE();

    INSERT INTO TelcoChurnQ3.Log (
        ErrorNumber, ErrorSeverity, ErrorState, 
        ErrorProcedure, ErrorLine, ErrorMessage, Operation
    )
    VALUES (
        ERROR_NUMBER(), ERROR_SEVERITY(), ERROR_STATE(), 
        ERROR_PROCEDURE(), ERROR_LINE(), ERROR_MESSAGE(), 'Dependent Tables Import'
    );
        
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
END CATCH;

-- Clean up the temp table
IF OBJECT_ID('tempdb..#TelcoData') IS NOT NULL
    DROP TABLE #TelcoData;

-- Disable xp_cmdshell
IF EXISTS (SELECT * FROM sys.configurations WHERE name = 'xp_cmdshell' AND value_in_use = 1)
BEGIN
    EXEC sp_configure 'xp_cmdshell', 0;
    RECONFIGURE;
END;

-- Disable Ad Hoc Queries for OPENROWSET
IF EXISTS (SELECT * FROM sys.configurations WHERE name = 'Ad Hoc Distributed Queries' AND value_in_use = 1)
BEGIN
    EXEC sp_configure 'Ad Hoc Distributed Queries', 0;
    RECONFIGURE;
END;

-- Terminate script execution
TerminateScriptExecution:
RETURN
GO