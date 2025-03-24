
/* Description: This script creates the schema for the CustomerChurnDB database */


SET NOCOUNT ON;
GO

-- Enable xp_cmdshell 
IF NOT EXISTS (SELECT * FROM sys.configurations 
				WHERE name = 'xp_cmdshell' AND value_in_use = 1)
BEGIN
    EXEC sp_configure 'show advanced options', 1;  
    RECONFIGURE;
    EXEC sp_configure 'xp_cmdshell', 1;
    RECONFIGURE;
END;
GO

-- Enable Ad Hoc Queries for OPENROWSET
IF NOT EXISTS (SELECT * FROM sys.configurations 
				WHERE name = 'Ad Hoc Distributed Queries' AND value_in_use = 1)
BEGIN
    EXEC sp_configure 'show advanced options', 1;  
    RECONFIGURE;
    EXEC sp_configure 'Ad Hoc Distributed Queries', 1;
    RECONFIGURE;
END;
GO

-- Create a dedicated database 
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'CustomerChurnDB')
BEGIN
	BEGIN TRY
		CREATE DATABASE CustomerChurnDB;
		PRINT 'Database CustomerChurnDB created successfully.';
	END TRY
	BEGIN CATCH
		PRINT 'Error creating database CustomerChurnDB: ' + ERROR_MESSAGE();
	END CATCH
END
ELSE
BEGIN
	PRINT 'Database CustomerChurnDB already exists or you do not have permission to create it.'
END
GO

USE CustomerChurnDB;
GO

-- Create a dedicated schema
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'Lakehouse')
BEGIN
	BEGIN TRY
		EXEC('CREATE SCHEMA Lakehouse');
		PRINT 'Schema Lakehouse created successfully.';
	END TRY
	BEGIN CATCH
		PRINT 'Error creating schema Lakehouse: ' + ERROR_MESSAGE();
	END CATCH
END
ELSE
BEGIN
	PRINT 'Schema Lakehouse already exists or you do not have permission to create it.'
	RETURN;
END
GO

-- Create Log table to track imports
IF EXISTS (SELECT * FROM sys.schemas WHERE name = 'Lakehouse')
BEGIN
    IF OBJECT_ID('Lakehouse.ImportLog') IS NOT NULL
        DROP TABLE Lakehouse.ImportLog;
    CREATE TABLE Lakehouse.ImportLog (
        LogID INT IDENTITY(1,1) PRIMARY KEY,
		ImportDate DATETIME DEFAULT GETDATE(),
		FileName NVARCHAR(255),
		TableName NVARCHAR(255),
		Status NVARCHAR(50),
		ErrorMessage NVARCHAR(MAX)
    );
    PRINT 'Table Lakehouse.ImportLog created successfully.';
END
ELSE
BEGIN
    PRINT 'Cannot create table: Schema Lakehouse does not exist.';
	RETURN;
END
GO

-- Batch-scoped variables declaration
DECLARE @ROOTDIR NVARCHAR(255) = '$(Param)\Datasets\';
DECLARE @FileName NVARCHAR(255);
DECLARE @FilePath NVARCHAR(500);
DECLARE @TableName NVARCHAR(255);
DECLARE @Cmd NVARCHAR(500);
DECLARE @ErrorMessage NVARCHAR(MAX);
DECLARE @Counter INT = 1;
DECLARE @ReturnCode INT;


-- Create temporary table to store file names
IF OBJECT_ID('tempdb..#FileList') IS NOT NULL
	DROP TABLE #FileList;
CREATE TABLE #FileList (ID INT IDENTITY(1,1), FileName NVARCHAR(255));

-- Fetch CSV files from the @ROOTDIR, 
-- but filters out the files generated during the data analysis steps  (in case of re-runs of the install script)
SET @Cmd = 'dir "' + @ROOTDIR + '*.csv" /b | findstr /v /c:"transformed" /c:"cleaned"';

INSERT INTO #FileList (FileName)
EXEC xp_cmdshell @Cmd;

-- Remove NULL values (sometimes xp_cmdshell adds extra rows)
DELETE FROM #FileList WHERE FileName IS NULL;

-- Create temporary mapping table of CSV files to table names
IF OBJECT_ID('tempdb..#FileTableMapping') IS NOT NULL
    DROP TABLE #FileTableMapping;
CREATE TABLE #FileTableMapping (FileName NVARCHAR(255), TableName NVARCHAR(255));
INSERT INTO #FileTableMapping (FileName, TableName)
	VALUES 
		('telco_customers.csv', 'Customers');

-- Get the total number of files
DECLARE @TotalFilesFound INT;
SET @TotalFilesFound = (SELECT COUNT(*) FROM #FileList);

-- Log 
IF EXISTS (SELECT * FROM #FileList 
	JOIN #FileTableMapping ON #FileList.FileName = #FileTableMapping.FileName)
	BEGIN
		DECLARE @MatchingTableNames NVARCHAR(MAX) = (SELECT STRING_AGG(TableName, ', ') FROM #FileTableMapping);
		INSERT INTO Lakehouse.ImportLog ([Status], [ErrorMessage]) 
			VALUES ('Success', 'Found source files');
	END
	ELSE
	BEGIN
		INSERT INTO Lakehouse.ImportLog ([Status], [ErrorMessage]) 
			VALUES ('Failed', 'No source files found');
			GOTO TerminateScriptExecution;
	END;




/* SUBROUTINE: Process source files */
WHILE @Counter <= @TotalFilesFound
BEGIN
    -- Get the file name currently being processed
    SELECT @FileName = FileName FROM #FileList WHERE ID = @Counter;

	-- Get the corresponding table name from the mapping
    SELECT @TableName = TableName FROM #FileTableMapping WHERE FileName = @FileName;

	

	/* SUBROUTINE: Create table */
	-- Only proceed if a mapping exists
	IF @TableName IS NOT NULL
	BEGIN TRY
		-- Construct the full file path
		SET @FilePath = @ROOTDIR + @FileName;

		-- Check if the file exists
		DECLARE @FileExists INT;  
		EXEC master.dbo.xp_fileexist @FilePath, @FileExists OUTPUT;
		IF ISNULL(@FileExists, 0) = 0
		BEGIN
			SET @ErrorMessage = 'File ' + @FilePath + ' does not exist';
			INSERT INTO Lakehouse.ImportLog (FileName, TableName, Status, ErrorMessage)
				VALUES (@FileName, @TableName, 'Failed', @ErrorMessage);
			CONTINUE;
		END

		-- Update log with file and table names; set status to 'Started'
		INSERT INTO Lakehouse.ImportLog (FileName, TableName, ImportDate, Status, ErrorMessage)	
			VALUES (@FileName, @TableName, GETDATE(), 'Started', NULL);

		-- Get column headers from CSV 
		IF OBJECT_ID('tempdb..#ColumnHeaders') IS NOT NULL
			DROP TABLE #ColumnHeaders;
		CREATE TABLE #ColumnHeaders (HeaderRow NVARCHAR(MAX));
		DECLARE @SQL NVARCHAR(MAX);
		SET @SQL = '
			BULK INSERT #ColumnHeaders 
			FROM ''' + @FilePath + '''
			WITH (
				FORMAT = ''CSV'',
				FIRSTROW = 1, 
				LASTROW = 1,
				FIELDTERMINATOR = '','',
				ROWTERMINATOR = ''\n'',
				TABLOCK
			);';
		EXEC sp_executesql @SQL;

		-- Get first row of entries from CSV  
		-- TODO: Sample more rows to improve data type inference
		IF OBJECT_ID('tempdb..#SampleData') IS NOT NULL
			DROP TABLE #SampleData;
		CREATE TABLE #SampleData ( DataRow NVARCHAR(MAX) );
		SET @SQL = '  
			BULK INSERT #SampleData 
			FROM ''' + @FilePath + '''
			WITH (
				FORMAT = ''CSV'',
				FIRSTROW = 2, 
				LASTROW = 2,
				FIELDTERMINATOR = '','',
				ROWTERMINATOR = ''\n'',
				TABLOCK
			);';
		EXEC sp_executesql @SQL;

		-- Get columns count for each row
		DECLARE @COL_COUNTER_H INT;
		SET @COL_COUNTER_H = (
			SELECT COUNT(*) FROM #ColumnHeaders CROSS APPLY STRING_SPLIT(#ColumnHeaders.HeaderRow, ',')
		);
		DECLARE @COL_COUNTER_E INT;
		SET @COL_COUNTER_E = (
			SELECT COUNT(*) FROM #SampleData CROSS APPLY STRING_SPLIT(#SampleData.DataRow, ',')
		);
		

		-- Declare a column pointer for each row
		DECLARE @ColName NVARCHAR(MAX);
		DECLARE col_cursor_H CURSOR LOCAL FORWARD_ONLY
			FOR SELECT TOP (@COL_COUNTER_H) value FROM #ColumnHeaders CROSS APPLY STRING_SPLIT(#ColumnHeaders.HeaderRow, ',');
		OPEN col_cursor_H;
		FETCH NEXT FROM col_cursor_H INTO @ColName;
		
		DECLARE @ColValue NVARCHAR(MAX);
		DECLARE col_cursor_E CURSOR LOCAL FORWARD_ONLY 
			FOR SELECT TOP (@COL_COUNTER_E) value FROM #SampleData CROSS APPLY STRING_SPLIT(#SampleData.DataRow, ',');
		OPEN col_cursor_E;
		FETCH NEXT FROM col_cursor_E INTO @ColValue;



		
		/* SUB-ROUTINE: Iterate over cols to infer data type */
		DECLARE @ColumnList NVARCHAR(MAX) = '';
		DECLARE @ColDataType NVARCHAR(MAX);
		DECLARE @CheckSQL NVARCHAR(MAX);
		
		WHILE @@FETCH_STATUS = 0
		BEGIN
		
			-- Col header format sanitation
			SET @ColName = REPLACE( TRIM(@ColName), ' ','_' );	

			-- Evaluate @ColValue type and cast corresponding @ColName accordingly
			IF EXISTS (
				SELECT TRY_CAST(@ColValue AS SMALLINT) WHERE TRY_CAST(@ColValue AS SMALLINT) IS NOT NULL
				) SET @ColDataType = 'SMALLINT'
			ELSE IF EXISTS (
				SELECT TRY_CAST(@ColValue AS INT) WHERE TRY_CAST(@ColValue AS INT) IS NOT NULL
				) SET @ColDataType = 'INT'
			ELSE IF EXISTS (
				SELECT TRY_CAST(@ColValue AS FLOAT) WHERE TRY_CAST(@ColValue AS FLOAT) IS NOT NULL
				) SET @ColDataType = 'DECIMAL(18,3)'
			ELSE IF EXISTS (
				SELECT TRY_CAST(@ColValue AS DATE) WHERE TRY_CAST(@ColValue AS DATE) IS NOT NULL
				) SET @ColDataType = 'DATE'
			ELSE 
				-- Default to NVARCHAR(MAX)
				SET @ColDataType = 'NVARCHAR(255)';

			-- Append column to create table statement 
			SET @ColumnList = @ColumnList + @ColName + ' ' + @ColDataType + ',';
		
	
			FETCH NEXT FROM col_cursor_H INTO @ColName;
			FETCH NEXT FROM col_cursor_E INTO @ColValue;
		
		END;
		CLOSE col_cursor_H; DEALLOCATE col_cursor_H;
		CLOSE col_cursor_E;	DEALLOCATE col_cursor_E;
		/* END OF SUBROUTINE: Iterate over cols to infer data types */
		
		
		
		-- Remove last comma
		SET @ColumnList = LEFT(@ColumnList, LEN(@ColumnList) - 1);


		-- Create table if it does not exist
		IF OBJECT_ID('Lakehouse.' + QUOTENAME(@TableName) ) IS NOT NULL
		BEGIN
			SET @SQL = 'DROP TABLE Lakehouse.'+ @TableName 
			EXEC sp_executesql @SQL;
		END;
		SET @SQL = 'CREATE TABLE Lakehouse.' + @TableName + ' (' + @ColumnList + ');';
		EXEC sp_executesql @SQL;

		-- Log table creation as successful
		UPDATE Lakehouse.ImportLog
			SET Status = 'Succcess', ErrorMessage = 'Table Created'
			WHERE FileName = @FileName AND TableName = @TableName;

	END TRY
	BEGIN CATCH
		SET @ErrorMessage = ERROR_MESSAGE();
		INSERT INTO Lakehouse.ImportLog ([FileName],[TableName],[Status], [ErrorMessage]) 
			VALUES (@FileName, @TableName, 'Failed', @ErrorMessage); 
	END CATCH;
	/* END OF SUBROUTINE: Create table*/



	/* SUBROUTINE: Dynamic Bulk Insert */
	BEGIN TRY
		BEGIN TRANSACTION;
		
		SET @SQL = '
			BULK INSERT ' + 'Lakehouse.' + @TableName + '
			FROM ''' + @FilePath + '''
			WITH (
				FORMAT = ''CSV'',
				FIRSTROW = 2,
				FIELDTERMINATOR = '','', 
				ROWTERMINATOR = ''\n'',
				TABLOCK
			);';
		EXEC sp_executesql @SQL;
		
		COMMIT TRANSACTION;
		
		-- Log import status as successful
		UPDATE Lakehouse.ImportLog
			SET Status = 'Success'
			WHERE FileName = @FileName AND TableName = @TableName;
	END TRY
	BEGIN CATCH
		IF @@TRANCOUNT > 0
			ROLLBACK TRANSACTION;

		DECLARE @ErrorNumber INT = ERROR_NUMBER();
		DECLARE @ErrorLine INT = ERROR_LINE();
		DECLARE @ErrorProc NVARCHAR(128) = ERROR_PROCEDURE();
		
		SET @ErrorMessage = 'Error ' + CAST(@ErrorNumber AS NVARCHAR) + 
							', Line ' + CAST(@ErrorLine AS NVARCHAR) + 
							': ' + ERROR_MESSAGE();
							
		UPDATE Lakehouse.ImportLog
			SET Status = 'Data Import Failed', 
				ErrorMessage = @ErrorMessage
			WHERE FileName = @FileName AND TableName = @TableName;
	END CATCH;
	/* END OF SUBROUTINE */


	-- Export bcp / format file with native data types to bulk import into Azure Synapse (warehousing)
	BEGIN TRY
		SET @Cmd = 'bcp CustomerChurnDB.Lakehouse.' + @TableName + 
				' out ' + @ROOTDIR + @TableName + '.bcp -T -n';
		EXEC @ReturnCode = xp_cmdshell @Cmd;
		
		IF @ReturnCode <> 0
			RAISERROR('BCP export failed with return code %d', 16, 1, @ReturnCode);
			
		SET @Cmd = 'bcp CustomerChurnDB.Lakehouse.' + @TableName + 
				' format nul -n -t, -f ' + @ROOTDIR + @TableName + '.fmt -T';
		EXEC @ReturnCode = xp_cmdshell @Cmd;
		
		IF @ReturnCode <> 0
			RAISERROR('BCP format export failed with return code %d', 16, 1, @ReturnCode);
	END TRY
	BEGIN CATCH
		INSERT INTO Lakehouse.ImportLog (FileName, TableName, Status, ErrorMessage)
			VALUES (@FileName, @TableName, 'BCP Export Failed', ERROR_MESSAGE());
	END CATCH;

	-- Increment the counter
	SET	@Counter += 1;
END;

-- Cleanup
DROP TABLE #FileTableMapping;
DROP TABLE #FileList;
/* END OF SUBROUTINE: Process source files */


-- Create a logical backup of the database
BEGIN TRY
    IF NOT EXISTS(SELECT * FROM sys.backup_devices WHERE name = 'CustomerChurnData') 
        EXEC sp_addumpdevice 'disk','CustomerChurnData',
             '$(Param)\Datasets\CustomersData_LOGICAL.bak';
             
    BACKUP DATABASE CustomerChurnDB TO CustomerChurnData;
    EXEC sp_dropdevice 'CustomerChurnData';
    
    -- Log successful backup
    INSERT INTO Lakehouse.ImportLog (Status, ErrorMessage)
        VALUES ('Success', 'Database backup completed');
END TRY
BEGIN CATCH
    INSERT INTO Lakehouse.ImportLog (Status, ErrorMessage)
        VALUES ('Backup Failed', ERROR_MESSAGE());
END CATCH


-- Disable xp_cmdshell
IF EXISTS (SELECT * FROM sys.configurations 
				WHERE name = 'xp_cmdshell' AND value_in_use = 1)
BEGIN
	EXEC sp_configure 'xp_cmdshell', 0;
	RECONFIGURE;
END;

-- Disable Ad Hoc Queries for OPENROWSET
IF EXISTS (SELECT * FROM sys.configurations 
				WHERE name = 'Ad Hoc Distributed Queries' AND value_in_use = 1)
BEGIN
	EXEC sp_configure 'Ad Hoc Distributed Queries', 0;
	RECONFIGURE;
END;

-- Terminate script execution
TerminateScriptExecution:
RETURN
GO
