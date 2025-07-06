
echo "=========================================="
echo "   CSV Validation Script"
echo "   Neural Network Project"
echo "=========================================="

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: File $1 not found"
        return 1
    else
        echo "✅ File $1 found"
        return 0
    fi
}

count_lines() {
    if [ -f "$1" ]; then
        lines=$(wc -l < "$1")
        echo "📊 $1: $lines lines"
    fi
}

show_header() {
    if [ -f "$1" ]; then
        echo "📋 Header of $1:"
        head -n 3 "$1" | sed 's/^/   /'
        echo ""
    fi
}

validate_csv() {
    if [ -f "$1" ]; then
        echo "🔍 Validating CSV format for $1..."
        
        first_line=$(head -n 1 "$1")
        if [[ $first_line == *","* ]]; then
            echo "   ✅ CSV delimiter found"
        else
            echo "   ⚠️  Warning: No comma delimiter in header"
        fi
        
        header_cols=$(head -n 1 "$1" | tr ',' '\n' | wc -l)
        data_cols=$(sed -n '2p' "$1" | tr ',' '\n' | wc -l)
        
        echo "   📊 Header columns: $header_cols"
        echo "   📊 Data columns: $data_cols"
        
        if [ "$header_cols" -eq "$data_cols" ]; then
            echo "   ✅ Column count consistent"
        else
            echo "   ❌ Column count mismatch"
        fi
        
        echo ""
    fi
}

validate_numeric() {
    if [ -f "$1" ]; then
        echo "🔢 Checking numeric content in $1..."
        
        tail -n +2 "$1" > temp_data.csv
        
        non_numeric=$(grep -E '[^0-9.,-]' temp_data.csv | wc -l)
        
        if [ "$non_numeric" -eq 0 ]; then
            echo "   ✅ All data appears to be numeric"
        else
            echo "   ⚠️  Warning: $non_numeric lines contain non-numeric data"
            echo "   Sample non-numeric lines:"
            grep -E '[^0-9.,-]' temp_data.csv | head -n 3 | sed 's/^/      /'
        fi
        
        rm -f temp_data.csv
        echo ""
    fi
}

validate_dataset() {
    echo "🗂️  Validating dataset: $1"
    echo "----------------------------------------"
    
    if check_file "$1"; then
        count_lines "$1"
        show_header "$1"
        validate_csv "$1"
        validate_numeric "$1"
    fi
    
    echo ""
}

generate_report() {
    echo "📈 VALIDATION REPORT"
    echo "===================="
    echo "Timestamp: $(date)"
    echo ""
    
    echo "CSV files found:"
    for file in *.csv; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "  📄 $file ($size)"
        fi
    done
    echo ""
    
    expected_files=("xor_data.csv" "circle_data.csv" "predictions.csv")
    
    echo "Expected files status:"
    for file in "${expected_files[@]}"; do
        if [ -f "$file" ]; then
            echo "  ✅ $file - Found"
        else
            echo "  ❌ $file - Missing"
        fi
    done
    echo ""
}

create_sample_data() {
    echo "🎲 Creating sample datasets for testing..."
    
    cat > sample_xor.csv << EOF
input1,input2,output
0,0,0
0,1,1
1,0,1
1,1,0
EOF
    
    cat > sample_circle.csv << EOF
x,y,inside_circle
-0.5,-0.5,1
0.5,0.5,1
-0.9,-0.9,0
0.9,0.9,0
0.0,0.0,1
EOF
    
    echo "✅ Sample datasets created:"
    echo "   📄 sample_xor.csv"
    echo "   📄 sample_circle.csv"
    echo ""
}

main() {
    case "${1:-validate}" in
        "validate")
            generate_report
            
            for file in *.csv; do
                if [ -f "$file" ]; then
                    validate_dataset "$file"
                fi
            done
            
            if [ ! -f "*.csv" ]; then
                echo "ℹ️  No CSV files found in current directory"
                echo "   Run with 'sample' to create test data"
            fi
            ;;
            
        "sample")
            create_sample_data
            
            echo "Now validating sample data..."
            validate_dataset "sample_xor.csv"
            validate_dataset "sample_circle.csv"
            ;;
            
        "clean")
            echo "🧹 Cleaning up CSV files..."
            rm -f *.csv
            echo "✅ All CSV files removed"
            ;;
            
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  validate (default) - Validate existing CSV files"
            echo "  sample            - Create sample CSV files and validate"
            echo "  clean             - Remove all CSV files"
            echo "  help              - Show this help message"
            ;;
            
        *)
            echo "❌ Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

if [ ! -f "main.cpp" ] && [ ! -f "../main.cpp" ]; then
    echo "⚠️  Warning: This script should be run from the project root directory"
    echo "   Current directory: $(pwd)"
    echo ""
fi

main "$1"

echo "=========================================="
echo "   Validation completed"
echo "=========================================="
