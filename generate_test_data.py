import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_messy_data(n_rows=1000):
    """Generate a messy dataset with various data quality issues."""
    
    # Base data structure
    data = {
        'customer_id': [],
        'name': [],
        'email': [],
        'phone': [],
        'age': [],
        'salary': [],
        'department': [],
        'hire_date': [],
        'performance_score': [],
        'address': [],
        'city': [],
        'state': [],
        'zip_code': [],
        'employee_status': [],
        'last_review_date': []
    }
    
    # Sample data pools
    first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Chris', 'Anna', 'Tom', 'Emma', 
                   'James', 'Mary', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']
    departments = ['Sales', 'Marketing', 'Engineering', 'HR', 'Finance', 'Operations', 'IT', 'Legal']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA']
    
    for i in range(n_rows):
        # Customer ID with various issues
        if i < 50:  # Some missing IDs
            customer_id = None
        elif i < 100:  # Some duplicates
            customer_id = f"CUST_{random.randint(1, 50):04d}"
        elif i < 150:  # Format inconsistencies
            customer_id = f"cust-{i:03d}"
        elif i < 200:  # Leading/trailing spaces
            customer_id = f"  CUST_{i:04d}  "
        else:
            customer_id = f"CUST_{i:04d}"
        
        # Name with various issues
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        if i < 30:  # Missing names
            name = None
        elif i < 60:  # Inconsistent formatting
            name = f"{first_name.lower()} {last_name.upper()}"
        elif i < 90:  # Extra spaces
            name = f"  {first_name}   {last_name}  "
        elif i < 120:  # Numbers in names (data entry errors)
            name = f"{first_name}2 {last_name}"
        elif i < 150:  # Special characters
            name = f"{first_name}@ {last_name}#"
        else:
            name = f"{first_name} {last_name}"
        
        # Email with various issues
        if i < 40:  # Missing emails
            email = None
        elif i < 80:  # Invalid format
            email = f"{first_name.lower()}.{last_name.lower()}"  # Missing @domain
        elif i < 120:  # Mixed case inconsistency
            email = f"{first_name.upper()}.{last_name.lower()}@COMPANY.COM"
        elif i < 160:  # Extra spaces
            email = f"  {first_name.lower()}.{last_name.lower()}@company.com  "
        elif i < 200:  # Invalid domains
            email = f"{first_name.lower()}.{last_name.lower()}@invalid"
        else:
            email = f"{first_name.lower()}.{last_name.lower()}@company.com"
        
        # Phone with various formats and issues
        if i < 50:  # Missing phones
            phone = None
        elif i < 100:  # Different formats
            phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif i < 150:  # Dots format
            phone = f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(1000, 9999)}"
        elif i < 200:  # No formatting
            phone = f"{random.randint(1000000000, 9999999999)}"
        elif i < 250:  # With country code
            phone = f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif i < 300:  # Invalid (too short)
            phone = f"{random.randint(100000, 999999)}"
        else:
            phone = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        
        # Age with various issues
        if i < 30:  # Missing ages
            age = None
        elif i < 60:  # Negative ages (error)
            age = -random.randint(1, 50)
        elif i < 90:  # Unrealistic ages
            age = random.randint(150, 200)
        elif i < 120:  # Zero age
            age = 0
        elif i < 150:  # String ages (should be numeric)
            age = f"{random.randint(20, 65)} years"
        else:
            age = random.randint(18, 70)
        
        # Salary with various issues
        if i < 40:  # Missing salaries
            salary = None
        elif i < 80:  # Negative salaries
            salary = -random.randint(30000, 100000)
        elif i < 120:  # String format with $ and commas
            salary = f"${random.randint(30000, 120000):,}"
        elif i < 160:  # Unrealistic high salaries
            salary = random.randint(1000000, 10000000)
        elif i < 200:  # Very low salaries (likely errors)
            salary = random.randint(1, 1000)
        else:
            salary = random.randint(30000, 120000)
        
        # Department with inconsistencies
        if i < 50:  # Missing departments
            department = None
        elif i < 100:  # Case inconsistencies
            dept = random.choice(departments)
            department = dept.lower()
        elif i < 150:  # Extra spaces
            department = f"  {random.choice(departments)}  "
        elif i < 200:  # Typos
            dept = random.choice(departments)
            if dept == 'Engineering':
                department = 'Enginering'  # Missing 'e'
            elif dept == 'Marketing':
                department = 'Markting'  # Missing 'e'
            else:
                department = dept
        else:
            department = random.choice(departments)
        
        # Hire date with various formats and issues
        if i < 40:  # Missing dates
            hire_date = None
        elif i < 80:  # Different date formats
            date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            hire_date = date.strftime("%m/%d/%Y")
        elif i < 120:  # Another format
            date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            hire_date = date.strftime("%d-%m-%Y")
        elif i < 160:  # ISO format
            date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            hire_date = date.strftime("%Y-%m-%d")
        elif i < 200:  # Invalid dates
            hire_date = "2024-13-45"  # Invalid month and day
        elif i < 240:  # Future dates (impossible hire dates)
            date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 365))
            hire_date = date.strftime("%Y-%m-%d")
        else:
            date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            hire_date = date.strftime("%Y-%m-%d")
        
        # Performance score with issues
        if i < 40:  # Missing scores
            performance_score = None
        elif i < 80:  # Out of range (should be 1-5)
            performance_score = random.randint(6, 10)
        elif i < 120:  # Negative scores
            performance_score = -random.randint(1, 5)
        elif i < 160:  # Decimal scores as strings
            performance_score = f"{random.uniform(1, 5):.1f}"
        else:
            performance_score = random.randint(1, 5)
        
        # Address with inconsistencies
        street_num = random.randint(1, 9999)
        street_names = ['Main St', 'Oak Ave', 'Park Rd', 'First St', 'Second Ave', 'Elm St', 'Maple Ave']
        street = random.choice(street_names)
        
        if i < 50:  # Missing addresses
            address = None
        elif i < 100:  # Inconsistent abbreviations
            address = f"{street_num} {street.replace('St', 'Street').replace('Ave', 'Avenue').replace('Rd', 'Road')}"
        elif i < 150:  # Extra spaces and punctuation
            address = f"  {street_num},  {street}.  "
        else:
            address = f"{street_num} {street}"
        
        # City with issues
        if i < 40:  # Missing cities
            city = None
        elif i < 80:  # Case issues
            city = random.choice(cities).lower()
        elif i < 120:  # Extra spaces
            city = f"  {random.choice(cities)}  "
        else:
            city = random.choice(cities)
        
        # State with issues
        if i < 40:  # Missing states
            state = None
        elif i < 80:  # Full state names instead of abbreviations
            state_map = {'NY': 'New York', 'CA': 'California', 'IL': 'Illinois', 'TX': 'Texas', 'AZ': 'Arizona', 'PA': 'Pennsylvania'}
            state_abbr = random.choice(states)
            state = state_map.get(state_abbr, state_abbr)
        elif i < 120:  # Lowercase
            state = random.choice(states).lower()
        else:
            state = random.choice(states)
        
        # ZIP code with various issues
        if i < 50:  # Missing ZIP codes
            zip_code = None
        elif i < 100:  # Wrong format (should be 5 digits)
            zip_code = random.randint(1000, 9999)  # 4 digits
        elif i < 150:  # ZIP+4 format
            zip_code = f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
        elif i < 200:  # String with leading zeros issues
            zip_code = f"0{random.randint(1000, 9999)}"
        else:
            zip_code = random.randint(10000, 99999)
        
        # Employee status with inconsistencies
        statuses = ['Active', 'Inactive', 'Terminated', 'On Leave']
        if i < 40:  # Missing status
            employee_status = None
        elif i < 80:  # Case issues
            employee_status = random.choice(statuses).lower()
        elif i < 120:  # Extra spaces
            employee_status = f"  {random.choice(statuses)}  "
        elif i < 160:  # Different terms for same status
            if random.choice(statuses) == 'Active':
                employee_status = 'Current'
            elif random.choice(statuses) == 'Terminated':
                employee_status = 'Fired'
            else:
                employee_status = random.choice(statuses)
        else:
            employee_status = random.choice(statuses)
        
        # Last review date with issues
        if i < 60:  # Missing review dates
            last_review_date = None
        elif i < 120:  # Different formats
            date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
            last_review_date = date.strftime("%m/%d/%Y")
        elif i < 180:  # Future review dates (impossible)
            date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 365))
            last_review_date = date.strftime("%Y-%m-%d")
        else:
            date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
            last_review_date = date.strftime("%Y-%m-%d")
        
        # Add to data
        data['customer_id'].append(customer_id)
        data['name'].append(name)
        data['email'].append(email)
        data['phone'].append(phone)
        data['age'].append(age)
        data['salary'].append(salary)
        data['department'].append(department)
        data['hire_date'].append(hire_date)
        data['performance_score'].append(performance_score)
        data['address'].append(address)
        data['city'].append(city)
        data['state'].append(state)
        data['zip_code'].append(zip_code)
        data['employee_status'].append(employee_status)
        data['last_review_date'].append(last_review_date)
    
    return pd.DataFrame(data)

def generate_clean_data(messy_df):
    """Generate the ideal clean version of the messy dataset."""
    clean_df = messy_df.copy()
    
    # Clean customer_id
    clean_df['customer_id'] = clean_df['customer_id'].fillna('')
    clean_df['customer_id'] = clean_df['customer_id'].astype(str).str.strip()
    clean_df['customer_id'] = clean_df['customer_id'].str.upper()
    clean_df['customer_id'] = clean_df['customer_id'].str.replace(r'[^A-Z0-9_]', '', regex=True)
    # Fill missing with generated IDs
    missing_mask = (clean_df['customer_id'] == '') | (clean_df['customer_id'] == 'NAN')
    for idx in clean_df[missing_mask].index:
        clean_df.loc[idx, 'customer_id'] = f"CUST_{idx:04d}"
    
    # Clean name
    clean_df['name'] = clean_df['name'].fillna('')
    clean_df['name'] = clean_df['name'].astype(str).str.strip()
    # Remove numbers and special characters
    clean_df['name'] = clean_df['name'].str.replace(r'[0-9@#$%^&*()+=\[\]{}|\\:";\'<>?,./]', '', regex=True)
    # Fix case - title case
    clean_df['name'] = clean_df['name'].str.title()
    # Remove extra spaces
    clean_df['name'] = clean_df['name'].str.replace(r'\s+', ' ', regex=True)
    # Remove empty names
    clean_df['name'] = clean_df['name'].replace('', None)
    
    # Clean email
    clean_df['email'] = clean_df['email'].fillna('')
    clean_df['email'] = clean_df['email'].astype(str).str.strip().str.lower()
    # Fix invalid emails
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    invalid_emails = ~clean_df['email'].str.match(email_pattern, na=False)
    clean_df.loc[invalid_emails, 'email'] = None
    
    # Clean phone - standardize to XXX-XXX-XXXX format
    clean_df['phone'] = clean_df['phone'].fillna('')
    clean_df['phone'] = clean_df['phone'].astype(str)
    # Extract digits only
    clean_df['phone'] = clean_df['phone'].str.replace(r'[^0-9]', '', regex=True)
    # Keep only 10-digit phones
    valid_phones = clean_df['phone'].str.len() == 10
    clean_df.loc[~valid_phones, 'phone'] = None
    # Format valid phones
    valid_phone_mask = clean_df['phone'].notna() & (clean_df['phone'].str.len() == 10)
    clean_df.loc[valid_phone_mask, 'phone'] = clean_df.loc[valid_phone_mask, 'phone'].str.replace(
        r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', regex=True
    )
    
    # Clean age
    clean_df['age'] = pd.to_numeric(clean_df['age'], errors='coerce')
    # Remove unrealistic ages
    clean_df.loc[(clean_df['age'] < 16) | (clean_df['age'] > 100), 'age'] = None
    
    # Clean salary
    clean_df['salary'] = clean_df['salary'].astype(str).str.replace(r'[$,]', '', regex=True)
    clean_df['salary'] = pd.to_numeric(clean_df['salary'], errors='coerce')
    # Remove unrealistic salaries
    clean_df.loc[(clean_df['salary'] < 15000) | (clean_df['salary'] > 500000), 'salary'] = None
    
    # Clean department
    clean_df['department'] = clean_df['department'].fillna('')
    clean_df['department'] = clean_df['department'].astype(str).str.strip().str.title()
    # Fix common typos
    clean_df['department'] = clean_df['department'].replace({
        'Enginering': 'Engineering',
        'Markting': 'Marketing',
        '': None
    })
    
    # Clean hire_date - standardize to YYYY-MM-DD
    clean_df['hire_date'] = pd.to_datetime(clean_df['hire_date'], errors='coerce', infer_datetime_format=True)
    # Remove future dates
    today = datetime.now()
    clean_df.loc[clean_df['hire_date'] > today, 'hire_date'] = None
    # Convert to string format
    clean_df['hire_date'] = clean_df['hire_date'].dt.strftime('%Y-%m-%d')
    
    # Clean performance_score
    clean_df['performance_score'] = pd.to_numeric(clean_df['performance_score'], errors='coerce')
    # Keep only scores between 1 and 5
    clean_df.loc[(clean_df['performance_score'] < 1) | (clean_df['performance_score'] > 5), 'performance_score'] = None
    
    # Clean address
    clean_df['address'] = clean_df['address'].fillna('')
    clean_df['address'] = clean_df['address'].astype(str).str.strip()
    # Remove extra punctuation and spaces
    clean_df['address'] = clean_df['address'].str.replace(r'[,.]', '', regex=True)
    clean_df['address'] = clean_df['address'].str.replace(r'\s+', ' ', regex=True)
    clean_df['address'] = clean_df['address'].replace('', None)
    
    # Clean city
    clean_df['city'] = clean_df['city'].fillna('')
    clean_df['city'] = clean_df['city'].astype(str).str.strip().str.title()
    clean_df['city'] = clean_df['city'].replace('', None)
    
    # Clean state - standardize to 2-letter abbreviations
    state_mapping = {
        'new york': 'NY',
        'california': 'CA',
        'illinois': 'IL',
        'texas': 'TX',
        'arizona': 'AZ',
        'pennsylvania': 'PA'
    }
    clean_df['state'] = clean_df['state'].fillna('')
    clean_df['state'] = clean_df['state'].astype(str).str.strip().str.upper()
    # Convert full names to abbreviations
    for full_name, abbr in state_mapping.items():
        clean_df['state'] = clean_df['state'].str.replace(full_name.upper(), abbr)
    clean_df['state'] = clean_df['state'].replace('', None)
    
    # Clean zip_code - standardize to 5 digits
    clean_df['zip_code'] = clean_df['zip_code'].astype(str).str.split('-').str[0]  # Remove ZIP+4
    clean_df['zip_code'] = clean_df['zip_code'].str.replace(r'[^0-9]', '', regex=True)
    # Pad with leading zeros if needed
    clean_df['zip_code'] = clean_df['zip_code'].str.zfill(5)
    # Keep only 5-digit ZIPs
    valid_zips = clean_df['zip_code'].str.len() == 5
    clean_df.loc[~valid_zips, 'zip_code'] = None
    
    # Clean employee_status
    status_mapping = {
        'current': 'Active',
        'fired': 'Terminated'
    }
    clean_df['employee_status'] = clean_df['employee_status'].fillna('')
    clean_df['employee_status'] = clean_df['employee_status'].astype(str).str.strip().str.title()
    # Apply mappings
    for old_status, new_status in status_mapping.items():
        clean_df['employee_status'] = clean_df['employee_status'].str.replace(old_status.title(), new_status)
    clean_df['employee_status'] = clean_df['employee_status'].replace('', None)
    
    # Clean last_review_date
    clean_df['last_review_date'] = pd.to_datetime(clean_df['last_review_date'], errors='coerce', infer_datetime_format=True)
    # Remove future dates
    clean_df.loc[clean_df['last_review_date'] > today, 'last_review_date'] = None
    # Convert to string format
    clean_df['last_review_date'] = clean_df['last_review_date'].dt.strftime('%Y-%m-%d')
    
    return clean_df

if __name__ == "__main__":
    print("Generating messy test dataset with 1000 rows...")
    
    # Generate messy data
    messy_df = generate_messy_data(1000)
    print(f"Messy dataset created: {messy_df.shape}")
    print(f"Missing values per column:")
    print(messy_df.isnull().sum())
    
    # Save messy data
    messy_df.to_csv('messy_test_data.csv', index=False)
    print("Saved: messy_test_data.csv")
    
    # Generate clean version
    print("\nGenerating clean version...")
    clean_df = generate_clean_data(messy_df)
    print(f"Clean dataset created: {clean_df.shape}")
    print(f"Missing values per column (after cleaning):")
    print(clean_df.isnull().sum())
    
    # Save clean data
    clean_df.to_csv('ideal_clean_data.csv', index=False)
    print("Saved: ideal_clean_data.csv")
    
    print("\nData quality summary:")
    print("MESSY DATA ISSUES INCLUDED:")
    print("- Missing values across all columns")
    print("- Inconsistent formatting (case, spaces, punctuation)")
    print("- Invalid data (negative ages, impossible dates)")
    print("- Format inconsistencies (phone numbers, emails, dates)")
    print("- Typos and data entry errors")
    print("- Duplicate and invalid IDs")
    print("- Out-of-range values")
    print("- Mixed data types in numeric columns")
    
    print("\nCLEAN DATA FEATURES:")
    print("- Standardized formatting across all columns")
    print("- Invalid data removed or corrected")
    print("- Consistent date formats (YYYY-MM-DD)")
    print("- Standardized phone format (XXX-XXX-XXXX)")
    print("- Proper case formatting for text fields")
    print("- Valid email addresses only")
    print("- Realistic value ranges")
    print("- Consistent categorical values")
    
    print(f"\nFiles generated successfully!")
    print(f"Test with: streamlit run advanced_data_cleaner.py")
