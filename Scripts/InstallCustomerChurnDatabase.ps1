<#
.SYNOPSIS
	Installs the Customer Churn Database by executing a series of SQL and Python scripts.

.DESCRIPTION
	This script installs the Customer Churn Database by executing a predefined list of SQL and Python scripts. 
	It checks for the necessary prerequisites such as the SqlServer module, Python, and required Python packages.
	The script can optionally drop the existing database if the -Force parameter is specified.

.PARAMETER ServerInstance
	The name of the SQL Server instance where the database will be installed. Defaults to the local machine name.

.PARAMETER Force
	If specified, the script will drop the existing database before installation.

.EXAMPLE
	.\InstallCustomerChurnDatabase.ps1 -ServerInstance $env:COMPUTERNAME -Force

	Installs the Customer Churn Database on the specified SQL Server instance and drops the existing database if it exists.

.NOTES
	- The script requires the SqlServer PowerShell module.
	- Python (version >=3.10) must be installed and available in the system PATH.
	- The script reads the required Python packages from a requirements.txt file located in the same directory.
	  If PATH is not properly set, the script will throw a bunch of warnings and install the required packages using pip.
	- To desinstall the database, you can use the following command:
		$Database = "CustomerChurnDB"
		Invoke-Sqlcmd -ServerInstance $ServerInstance -Query "ALTER DATABASE [$Database] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [$Database]"
#>

[CmdletBinding()]
param(
	[Parameter(Mandatory = $true)]
	[ValidateNotNull()]
	[ValidateNotNullOrEmpty()]   
	[string]$ServerInstance = $env:COMPUTERNAME,

	[Parameter(Mandatory = $false)]
    [switch]$Force
)

process {
	
	# Define relative path
	$RootDir = Resolve-Path "..\"

	# Define the target database
	$Database = "CustomerChurnDB"

	# List of scripts to execute in this order
	$Files = @(
		"SQLQuery_ImportRawData.sql", 
		"SQLQuery_3NF.sql", 
		"CohortAnalysis.py", 
		#"PredictiveModeling.py"
		"SQLQuery_ImportTransformedData.sql"
	)

	# Initialize error count
	$errorCount = 0

	# Start the installation
	$timeStamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
	Write-Host "`nInstallation starting...`n" -ForegroundColor Green
	Write-Host "Checking required modules and packages" -ForegroundColor Yellow

	# Check if the SqlServer module is installed; if not, install it
	if (-not (Get-Module -ListAvailable -Name SqlServer)) {
		Write-Host "`nSqlServer module is not installed. Installing the module..."
		try {
			Install-Module -Name SqlServer -Force -Scope CurrentUser
			Write-Host "SqlServer module installed successfully."
		}
		catch {
			Write-Error "Failed to install the SqlServer module. Please install it manually."
			return
		}
	}
	else {
		$version = (Get-Module -ListAvailable -Name SqlServer).Version
		Write-Host "SqlServer module is already installed (version: $version)." 
	}

	# Check if the SqlServer module is loaded; if not, load it
	if (-not (Get-Module -Name SqlServer)) {
		Write-Host "`nLoading the SqlServer module..."
		try {
			Import-Module -Name SqlServer
			Write-Host "SqlServer module loaded successfully."
		}
		catch {
			Write-Error "Failed to load the SqlServer module. Please load it manually."
			return
		}
	}

	# Check if Python is installed
	if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
		Write-Error "`nPython is not installed. Please install Python (version >=3.10) before running this script."
		return
	}

	# Check if pip is installed
	if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
		Write-Error "`npip is not installed. Please install pip before running this script."
		return
	}

	# Check if required Python packages are installed
	$requiredPythonPackages = (Get-Content "./requirements.txt" -Raw) -split '\r?\n' | Where-Object { $_ -match '\S' }
	if (-not $requiredPythonPackages) {
		Write-Warning "requirements.txt file not found. Installation will proceed without installing any Python packages."
	}
	else {
		foreach ($package in $requiredPythonPackages) {
			# Extract just the package name without version specifiers
			$packageName = $package -replace '([a-zA-Z0-9_\-\.]+).*', '$1'
			
			# Check if it's installed
			$pipShowResult = & python -m pip show $packageName 2>$null
			
			if (-not $pipShowResult) {
				Write-Warning "Required Python package '$packageName' is not installed. Installing now..."
				python -m pip install $package
			}
			else {
				$installedVersion = ($pipShowResult | Select-String -Pattern "Version: (.+)").Matches.Groups[1].Value
				Write-Host "Package $packageName is already installed (version: $installedVersion)" 
			}
		}
	}


	# Loop through the list of files and execute
	foreach ($File in $Files) {
	
		Write-Host "`nStarting execution of $File" -ForegroundColor Yellow
		
		# Resolve path to the file
		$FilePath = Resolve-Path $File -ErrorAction SilentlyContinue
		if (-not $FilePath) {
			$FilePath = Join-Path -Path (Get-Location) -ChildPath $File
		}
		
		# Check if file exists
		if (-Not (Test-Path $FilePath)) {
			Write-Error "`n$File file not found"
			exit
		}
		
		
		if ($File -like "*.sql") {
			# Execute SQL scripts
			# Check if the SQL Server instance is reachable
			try {
				$Server = Get-Item "SQLServer:\SQL\$ServerInstance"
			} 
			catch {
				Write-Error "SQL Server instance $ServerInstance not found"
				return
			}
			# Check if the database exists
			try {
				$dbExists = Invoke-Sqlcmd -ServerInstance $ServerInstance -Query "SELECT name FROM sys.databases WHERE name = '$Database'"
			}
			catch {
				$dbExists = $null
			}
			
			# Drop the database if it exists and the Force flag is specified
            if ($dbExists -and $Force) {
                Write-Host "Database $Database exists. Force flag specified, dropping database..." 
                Invoke-Sqlcmd -ServerInstance $ServerInstance -Query "ALTER DATABASE [$Database] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [$Database]"
                $dbExists = $null
            }

			# Execute SQL script against master if DB doesn't exist, otherwise use the specified database
			try {
				if ($dbExists -eq $null) {
					Write-Host "Database $Database doesn't exist yet. Running script against master..." 
					Invoke-Sqlcmd -ServerInstance $ServerInstance -Database "master" -InputFile $FilePath -Variable @{Param=$RootDir} 
				} 
				else {
					Write-Host "Running script against existing database $Database..." 
					try {
						Invoke-Sqlcmd -ServerInstance $ServerInstance -Database $Database -InputFile $FilePath -Variable @{Param=$RootDir}
					}
					catch {
						Write-Warning "Failed to connect to existing database."
						# Try running against master instead since we can't access the DB
						try {
							Write-Host "Attempting to run against master database instead..." -ForegroundColor Yellow
							Invoke-Sqlcmd -ServerInstance $ServerInstance -Database "master" -InputFile $FilePath -Variable @{Param=$RootDir}
						}
						catch {
							Write-Error "Failed to execute script against master as well: $_"
							$errorCount++
							continue
						}
					}
				}
			}
			catch {
				Write-Error "Error executing SQL script: $_"
				$errorCount++
				continue
			}

		} elseif ($File -like "*.py") {
			# Execute the Python scripts
			try {
				$PythonPath = (Get-Command python).Source
				$PythonCommand = "$PythonPath $FilePath"
				Write-Host "Executing Python script: $PythonCommand"
				Invoke-Expression $PythonCommand
			}
			catch {
				Write-Error "Error executing Python script: $_"
				$errorCount++
				continue
			}
		} else {
			Write-Error "Unsupported file type: $File"
			exit
		}

		Write-Host "Completed execution of $File" 
	}

	# Display installation summary
	Write-Host "`nInstallation Summary`n--------------------" -ForegroundColor Green
	Write-Host "Database: $Database on $ServerInstance"
	Write-Host "$($Files.Count) scripts executed"
	write-host "Took $(New-TimeSpan -Start $timeStamp -End (Get-Date)) to complete"
	
	if ($errorCount -gt 0) {
    	Write-Host "Installation terminated with $errorCount errors`n"
	}
	else {
		Write-Host "Installation terminated successfully`n" -ForegroundColor Yellow
	}

}
