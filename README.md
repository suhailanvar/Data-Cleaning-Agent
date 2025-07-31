# 🧹📊 Advanced Data Cleaning & Validation Agent

An intelligent, AI-powered data cleaning application that analyzes your CSV datasets, identifies quality issues, and generates custom Python code to clean your data. Built with Streamlit and powered by Large Language Models (LLMs).

## ✨ Features

### 🔍 **Intelligent Analysis**
- **Multi-Sample Analysis**: Extracts diverse samples from your dataset for comprehensive quality assessment
- **Combined Dataset Approach**: Analyzes all samples together for unified insights
- **Column-Type Detection**: Automatically identifies email, phone, date, numeric, categorical, and other data types
- **Pattern Recognition**: Detects missing values, format inconsistencies, invalid data, and duplicates

### 🎯 **Interactive Strategy Selection**
- **Simple Mode**: Apply global cleaning rules across all columns
- **Advanced Mode**: Column-wise customization with intelligent recommendations
- **Tabbed Interface**: Separate controls for missing values, validation, and format strategies
- **Smart Validation**: Prevents inappropriate strategy combinations with real-time warnings

### 🤖 **AI-Powered Code Generation**
- **Custom Cleaning Code**: Generates Python functions tailored to your specific data issues
- **User Strategy Implementation**: Incorporates your selected cleaning preferences into the generated code
- **Safe Execution Environment**: Runs cleaning code in controlled environment with detailed logging
- **Real-time Streaming**: Watch the AI generate and execute code in real-time

### 📈 **Comprehensive Validation**
- **Before/After Comparison**: Validates cleaning effectiveness by comparing original vs cleaned datasets
- **Quality Metrics**: Quantifies improvements in data quality
- **Issue Tracking**: Identifies resolved issues and any remaining problems
- **Data Integrity Checks**: Ensures no critical data was lost during cleaning

### 🔄 **Dual LLM Support**
- **OpenRouter Integration**: Access to cloud-based models including GPT-4, Claude, Gemini, and more
- **Local Ollama Support**: Use local models for privacy and offline processing
- **Streaming Responses**: Real-time output for better user experience
- **Model Flexibility**: Switch between different models based on your needs

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/suhailanvar/Data-Cleaning-Agent.git
   cd Data-Cleaning-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional for OpenRouter)
   ```bash
   # Create .env file
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run advanced_data_cleaner.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 🛠️ Usage Guide

### Step 1: Configure LLM Provider
- Choose between **OpenRouter** (cloud) or **Ollama** (local)
- For OpenRouter: Enter your API key or set it in the `.env` file
- For Ollama: Ensure Ollama is running locally (`ollama serve`)

### Step 2: Upload Your Data
- Upload a CSV file using the file uploader
- The app will automatically load and analyze the dataset structure

### Step 3: Data Analysis
- Configure analysis settings (number of samples, sample size)
- Click "🔍 Analyze Data Quality"
- Watch as the AI analyzes your data and identifies quality issues

### Step 4: Strategy Selection
- **Simple Mode**: Choose global strategies for missing values, validation, and outliers
- **Advanced Mode**: Customize strategies for each problematic column
  - Missing Values tab: Handle null/empty values per column
  - Data Validation tab: Set validation rules per column
  - Format Issues tab: Standardize formats per column

### Step 5: Code Generation & Execution
- Click "🧹 Generate & Execute Cleaning Code"
- Watch the AI generate Python code based on your strategies
- The code will be automatically executed on your dataset

### Step 6: Validation & Results
- Review the validation results comparing before/after datasets
- Download the cleaned data as CSV
- View detailed logging and quality metrics

## 📊 Supported Data Types & Issues

### Data Types
- **Email**: Validates format, removes duplicates, standardizes case
- **Phone**: Standardizes formats, validates length, removes invalid characters
- **Date**: Standardizes formats, handles invalid dates, timezone conversion
- **Numeric**: Handles outliers, validates ranges, type conversion
- **Categorical**: Standardizes categories, handles inconsistent naming
- **Text**: Removes extra whitespace, standardizes case, handles encoding
- **Identifiers**: Validates uniqueness, generates missing IDs

### Common Issues Handled
- Missing values (nulls, empty strings, placeholders)
- Format inconsistencies (date formats, phone formats, email case)
- Invalid data (malformed emails, impossible dates, negative values)
- Duplicates (exact and fuzzy matching)
- Outliers (statistical and domain-specific)
- Encoding issues (special characters, unicode)
- Data type mismatches

## 🔧 Configuration Options

### Analysis Settings
- **Number of Samples**: 2-5 samples for analysis (more = better analysis, slower processing)
- **Sample Size**: 10-100 rows per sample
- **Streaming**: Enable real-time output during processing

### LLM Provider Settings

#### OpenRouter
- **API Key**: Your OpenRouter API key (secure, not logged)
- **Models**: 15+ available models including:
  - Free: Gemma, Llama, Phi-3, Zephyr
  - Premium: GPT-4, Claude, Gemini Pro
- **Timeout**: 300 seconds for large datasets

#### Ollama
- **Base URL**: Default `http://localhost:11434`
- **Models**: Any locally installed Ollama model
- **Timeout**: 3000 seconds for large datasets

## 📁 Project Structure

```
Data-Cleaning-Agent/
├── advanced_data_cleaner.py      # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .env                         # Environment variables (create yourself)
├── sample_data.csv              # Example dataset for testing
├── large_sample_data.csv        # Larger test dataset
├── messy_test_data.csv          # Dataset with intentional quality issues
├── generate_test_data.py        # Script to generate test datasets
└── test_files/                  # Additional testing utilities
    ├── test_*.py               # Various test scripts
    └── TEST_DATA_README.md     # Documentation for test data
```

## 🔍 Example Workflow

```python
# 1. Upload dataset with quality issues
Dataset: customer_data.csv (1000 rows, 15 columns)
Issues found: Missing emails, inconsistent phone formats, invalid dates

# 2. Analysis Results
- Email column: 15% missing, mixed case formatting
- Phone column: 22% missing, multiple formats (555-1234, (555) 123-4567)
- Join_Date column: 3% invalid dates ("invalid_date", future dates)

# 3. Strategy Selection (Advanced Mode)
- Email: "Keep missing emails as null, standardize to lowercase"
- Phone: "Standardize to format +1-XXX-XXX-XXXX, remove invalid"
- Join_Date: "Remove invalid dates, standardize to YYYY-MM-DD"

# 4. Generated Code
def clean_dataset(df):
    # Handle email column
    df['Email'] = df['Email'].str.lower()
    df['Email'] = df['Email'].replace('', pd.NA)
    
    # Standardize phone numbers
    df['Phone'] = df['Phone'].str.replace(r'[^\d]', '', regex=True)
    df['Phone'] = df['Phone'].apply(lambda x: f"+1-{x[0:3]}-{x[3:6]}-{x[6:10]}" if len(x) == 10 else pd.NA)
    
    # Clean join dates
    df['Join_Date'] = pd.to_datetime(df['Join_Date'], errors='coerce')
    df = df.dropna(subset=['Join_Date'])
    
    return df

# 5. Results
Original: 1000 rows → Cleaned: 976 rows
Quality improvement: EXCELLENT
Issues resolved: 95% of format inconsistencies fixed
```

## 🚨 Troubleshooting

### Common Issues

1. **"Could not connect to Ollama"**
   - Ensure Ollama is running: `ollama serve`
   - Check if port 11434 is available
   - Try restarting Ollama service

2. **"Invalid API key"**
   - Verify your OpenRouter API key is correct
   - Check if the key has sufficient credits
   - Ensure the key is properly set in `.env` file

3. **"Request timed out"**
   - Reduce sample size or number of samples
   - Try a faster model (e.g., smaller Ollama models)
   - Check your internet connection for OpenRouter

4. **"Could not extract Python code"**
   - The LLM may have generated malformed code
   - Try a different model or simplify your strategy selection
   - Check the logs for detailed error information

5. **Strategy selection causes UI reset**
   - This should be fixed in the current version
   - If it persists, try refreshing the page and re-uploading data

### Debug Mode
Enable debug information in the sidebar to see:
- Current session state
- Analysis results
- Generated prompts
- LLM responses
- Error details

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/Data-Cleaning-Agent.git
cd Data-Cleaning-Agent

# Install development dependencies
pip install -r requirements.txt
pip install streamlit pytest black flake8

# Run tests
python -m pytest test_*.py

# Format code
black advanced_data_cleaner.py

# Run the app in development mode
streamlit run advanced_data_cleaner.py --server.runOnSave true
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **OpenRouter** for cloud LLM access
- **Ollama** for local LLM capabilities
- **Pandas** for data manipulation
- The open-source community for continuous inspiration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/suhailanvar/Data-Cleaning-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/suhailanvar/Data-Cleaning-Agent/discussions)
- **Email**: suhailanvar@example.com

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for Excel and JSON files
- [ ] Advanced statistical outlier detection
- [ ] Data profiling and quality scoring
- [ ] Automated data type inference
- [ ] Integration with data catalogs
- [ ] Batch processing for multiple files
- [ ] Custom validation rules engine
- [ ] Data lineage tracking
- [ ] API endpoint for programmatic access
- [ ] Docker containerization

### Performance Improvements
- [ ] Parallel processing for large datasets
- [ ] Incremental cleaning for streaming data
- [ ] Memory optimization for large files
- [ ] Caching for repeated analyses

---

⭐ **Star this repository if you find it helpful!**

Built with ❤️ by [Suhail Anvar](https://github.com/suhailanvar)
