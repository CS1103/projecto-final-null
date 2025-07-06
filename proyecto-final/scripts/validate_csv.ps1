

param(
    [string]$Command = "validate"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   CSV Validation Script (PowerShell)"     -ForegroundColor Cyan
Write-Host "   Neural Network Project"                  -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Check-File {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "✅ File $FilePath found" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Error: File $FilePath not found" -ForegroundColor Red
        return $false
    }
}

function Count-Lines {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        $lines = (Get-Content $FilePath | Measure-Object -Line).Lines
        Write-Host "📊 $FilePath`: $lines lines" -ForegroundColor Yellow
    }
}

function Show-Header {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "📋 Header of $FilePath`:" -ForegroundColor Magenta
        $header = Get-Content $FilePath -Head 3
        foreach ($line in $header) {
            Write-Host "   $line" -ForegroundColor Gray
        }
        Write-Host ""
    }
}

function Validate-CSV {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "🔍 Validating CSV format for $FilePath..." -ForegroundColor Blue
        
        $firstLine = Get-Content $FilePath -Head 1
        if ($firstLine -like "*,*") {
            Write-Host "   ✅ CSV delimiter found" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Warning: No comma delimiter in header" -ForegroundColor Yellow
        }
        
        $headerCols = ($firstLine -split ",").Count
        $secondLine = Get-Content $FilePath | Select-Object -Skip 1 -First 1
        $dataCols = ($secondLine -split ",").Count
        
        Write-Host "   📊 Header columns: $headerCols" -ForegroundColor Yellow
        Write-Host "   📊 Data columns: $dataCols" -ForegroundColor Yellow
        
        if ($headerCols -eq $dataCols) {
            Write-Host "   ✅ Column count consistent" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Column count mismatch" -ForegroundColor Red
        }
        
        Write-Host ""
    }
}

function Validate-Numeric {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "🔢 Checking numeric content in $FilePath..." -ForegroundColor Blue
        
        $dataLines = Get-Content $FilePath | Select-Object -Skip 1
        
        $nonNumericLines = $dataLines | Where-Object { $_ -match '[^0-9.,-]' }
        $nonNumericCount = ($nonNumericLines | Measure-Object).Count
        
        if ($nonNumericCount -eq 0) {
            Write-Host "   ✅ All data appears to be numeric" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Warning: $nonNumericCount lines contain non-numeric data" -ForegroundColor Yellow
            Write-Host "   Sample non-numeric lines:" -ForegroundColor Yellow
            $nonNumericLines | Select-Object -First 3 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor Gray
            }
        }
        
        Write-Host ""
    }
}

function Validate-Dataset {
    param([string]$FilePath)
    
    Write-Host "🗂️  Validating dataset: $FilePath" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    
    if (Check-File $FilePath) {
        Count-Lines $FilePath
        Show-Header $FilePath
        Validate-CSV $FilePath
        Validate-Numeric $FilePath
    }
    
    Write-Host ""
}

function Generate-Report {
    Write-Host "📈 VALIDATION REPORT" -ForegroundColor Cyan
    Write-Host "====================" -ForegroundColor Cyan
    Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "CSV files found:" -ForegroundColor Magenta
    $csvFiles = Get-ChildItem -Filter "*.csv"
    foreach ($file in $csvFiles) {
        $size = [math]::Round($file.Length / 1KB, 2)
        Write-Host "  📄 $($file.Name) ($size KB)" -ForegroundColor Gray
    }
    
    if ($csvFiles.Count -eq 0) {
        Write-Host "  No CSV files found" -ForegroundColor Gray
    }
    Write-Host ""
    
    $expectedFiles = @("xor_data.csv", "circle_data.csv", "predictions.csv")
    
    Write-Host "Expected files status:" -ForegroundColor Magenta
    foreach ($file in $expectedFiles) {
        if (Test-Path $file) {
            Write-Host "  ✅ $file - Found" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $file - Missing" -ForegroundColor Red
        }
    }
    Write-Host ""
}

function Create-SampleData {
    Write-Host "🎲 Creating sample datasets for testing..." -ForegroundColor Blue
    
    $xorContent = @"
input1,input2,output
0,0,0
0,1,1
1,0,1
1,1,0
"@
    
    $xorContent | Out-File -FilePath "sample_xor.csv" -Encoding UTF8
    
    $circleContent = @"
x,y,inside_circle
-0.5,-0.5,1
0.5,0.5,1
-0.9,-0.9,0
0.9,0.9,0
0.0,0.0,1
"@
    
    $circleContent | Out-File -FilePath "sample_circle.csv" -Encoding UTF8
    
    Write-Host "✅ Sample datasets created:" -ForegroundColor Green
    Write-Host "   📄 sample_xor.csv" -ForegroundColor Gray
    Write-Host "   📄 sample_circle.csv" -ForegroundColor Gray
    Write-Host ""
}

switch ($Command.ToLower()) {
    "validate" {
        Generate-Report
        
        $csvFiles = Get-ChildItem -Filter "*.csv"
        
        if ($csvFiles.Count -gt 0) {
            foreach ($file in $csvFiles) {
                Validate-Dataset $file.Name
            }
        } else {
            Write-Host "ℹ️  No CSV files found in current directory" -ForegroundColor Blue
            Write-Host "   Run with 'sample' parameter to create test data" -ForegroundColor Blue
        }
    }
    
    "sample" {
        Create-SampleData
        
        Write-Host "Now validating sample data..." -ForegroundColor Blue
        Validate-Dataset "sample_xor.csv"
        Validate-Dataset "sample_circle.csv"
    }
    
    "clean" {
        Write-Host "🧹 Cleaning up CSV files..." -ForegroundColor Blue
        Remove-Item -Path "*.csv" -Force -ErrorAction SilentlyContinue
        Write-Host "✅ All CSV files removed" -ForegroundColor Green
    }
    
    "help" {
        Write-Host "Usage: .\validate_csv.ps1 [-Command <command>]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  validate (default) - Validate existing CSV files" -ForegroundColor Gray
        Write-Host "  sample            - Create sample CSV files and validate" -ForegroundColor Gray
        Write-Host "  clean             - Remove all CSV files" -ForegroundColor Gray
        Write-Host "  help              - Show this help message" -ForegroundColor Gray
    }
    
    default {
        Write-Host "❌ Unknown command: $Command" -ForegroundColor Red
        Write-Host "Use '.\validate_csv.ps1 -Command help' for usage information" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path "main.cpp") -and -not (Test-Path "..\main.cpp")) {
    Write-Host "⚠️  Warning: This script should be run from the project root directory" -ForegroundColor Yellow
    Write-Host "   Current directory: $(Get-Location)" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Validation completed" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
