# Test Data Generation Summary

## Files Created

### 1. `messy_test_data.csv` (1000 rows)
A comprehensive test dataset with **15 columns** containing various data quality issues:

#### Data Quality Issues Included:

**Customer IDs:**
- Missing values (50 rows)
- Duplicate IDs (rows 50-100)
- Format inconsistencies: "cust-XXX" vs "CUST_XXXX"
- Leading/trailing spaces: "  CUST_0001  "

**Names:**
- Missing values (30 rows)
- Case inconsistencies: "john SMITH", "MARY jones"
- Extra spaces: "  John   Smith  "
- Invalid characters: "John2 Smith", "Mary@ Jones#"

**Emails:**
- Missing values (40 rows)
- Invalid formats: "john.smith" (no @domain)
- Case inconsistencies: "JOHN.smith@COMPANY.COM"
- Extra spaces: "  john.smith@company.com  "
- Invalid domains: "john.smith@invalid"

**Phone Numbers:**
- Missing values (50 rows)
- Multiple formats: "(555) 123-4567", "555.123.4567", "5551234567"
- Invalid lengths: short numbers
- Country codes: "+1-555-123-4567"

**Ages:**
- Missing values (30 rows)
- Negative ages: -25, -38
- Unrealistic ages: 150, 200
- Zero ages
- String formats: "25 years"

**Salaries:**
- Missing values (40 rows)
- Negative values: -50000
- String formats with currency: "$65,000"
- Unrealistic values: $10,000,000
- Very low values: $100

**Departments:**
- Missing values (50 rows)
- Case issues: "sales", "MARKETING"
- Extra spaces: "  HR  "
- Typos: "Enginering", "Markting"

**Hire Dates:**
- Missing values (40 rows)
- Multiple formats: "MM/DD/YYYY", "DD-MM-YYYY", "YYYY-MM-DD"
- Invalid dates: "2024-13-45"
- Future dates (impossible)

**Performance Scores:**
- Missing values (40 rows)
- Out of range: 6-10 (should be 1-5)
- Negative scores: -3
- String formats: "4.5"

**Addresses:**
- Missing values (50 rows)
- Inconsistent abbreviations: "Street" vs "St"
- Extra punctuation: "123, Main St."

**Cities & States:**
- Missing values (40 rows)
- Case issues: "new york", "CALIFORNIA"
- Extra spaces
- Full state names vs abbreviations

**ZIP Codes:**
- Missing values (50 rows)
- Wrong lengths: 4-digit codes
- ZIP+4 format: "12345-6789"
- Leading zero issues

**Employee Status:**
- Missing values (40 rows)
- Case issues: "active", "TERMINATED"
- Different terms: "Current" vs "Active", "Fired" vs "Terminated"

**Review Dates:**
- Missing values (60 rows)
- Different formats
- Future dates (impossible)

### 2. `ideal_clean_data.csv` (1000 rows)
The same dataset after comprehensive cleaning:

#### Cleaning Applied:

- **Standardized formats** across all columns
- **Removed invalid data** (negative ages, impossible dates)
- **Consistent case formatting** (Title Case for names, lowercase for emails)
- **Standardized phone format**: XXX-XXX-XXXX
- **Consistent date format**: YYYY-MM-DD
- **Proper data types**: numeric columns converted appropriately
- **Removed outliers** and unrealistic values
- **Fixed typos** and data entry errors
- **Standardized categorical values**

### 3. Supporting Scripts

- **`generate_test_data.py`**: Main script to generate both datasets
- **`compare_data.py`**: Comparison analysis between messy and clean data
- **`show_examples.py`**: Shows specific examples of data quality issues

## Usage

1. **Test the Advanced Data Cleaner:**
   ```bash
   streamlit run advanced_data_cleaner.py
   ```

2. **Upload `messy_test_data.csv`** to test the cleaning capabilities

3. **Compare results** with `ideal_clean_data.csv` to validate cleaning effectiveness

## Data Statistics

**Messy Data:**
- **Rows**: 1000
- **Columns**: 15
- **Missing Values**: 650+ across all columns
- **Data Types**: Mostly strings/objects due to formatting issues

**Clean Data:**
- **Rows**: 1000
- **Columns**: 15
- **Missing Values**: Properly handled (some legitimately missing)
- **Data Types**: Appropriate numeric types where applicable

## Perfect for Testing

This dataset is ideal for testing data cleaning algorithms because it includes:

✅ **Real-world complexity** - Multiple issue types per column
✅ **Scalable size** - 1000 rows for performance testing
✅ **Comprehensive coverage** - All common data quality issues
✅ **Validation baseline** - Clean version for comparison
✅ **Diverse patterns** - Issues distributed throughout dataset
✅ **Production scenarios** - Realistic business data structure

The dataset will thoroughly test your advanced data cleaner's ability to:
- Identify diverse data quality issues
- Generate appropriate cleaning code
- Handle edge cases and complex patterns
- Validate cleaning effectiveness
