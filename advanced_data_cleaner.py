from click import prompt
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import io
import os
import numpy as np
import random
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---- Logging Configuration ----
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_data_cleaner.log'),
        logging.StreamHandler()
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# ---- Config ----
# API key from environment variables or Streamlit secrets
try:
    # Try Streamlit secrets first (for cloud deployment)
    DEFAULT_OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
except:
    # Fallback to environment variables (for local development)
    DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available OpenRouter models (recommendations - not restrictive)
OPENROUTER_MODELS = [
    # Free models
    "google/gemma-3-27b-it:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    # Popular paid models
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-pro-1.5",
    "mistralai/mistral-7b-instruct",
    "qwen/qwen-2.5-7b-instruct"
]

# ---- Helper Functions ----
def log_step(step_name, details, log_container):
    """Add a step to the processing log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    logger.info(f"{step_name}: {details}")
    with log_container.container():
        st.markdown(f"**🕐 {timestamp} - {step_name}**")
        st.markdown(f"   {details}")

def get_available_ollama_models():
    """Get list of available Ollama models."""
    logger.debug("Attempting to fetch available Ollama models")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            logger.info(f"Successfully retrieved {len(models)} Ollama models: {models}")
            return models
        else:
            logger.warning(f"Failed to fetch Ollama models: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

def call_openrouter(prompt, model_name, api_key, stream=False):
    """Call OpenRouter API with error handling and optional streaming."""
    logger.debug(f"Calling OpenRouter API with model: {model_name}, stream: {stream}")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        
        logger.debug(f"Sending request to OpenRouter with {len(prompt)} character prompt")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=300, stream=stream)
        response.raise_for_status()
        
        if stream:
            logger.debug("Returning streaming response object")
            return response  # Return response object for streaming
        else:
            result = response.json()
            logger.info("Successfully received non-streaming response from OpenRouter")
            return result['choices'][0]['message']['content']
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error with OpenRouter API: {e}")
        raise Exception("Could not connect to OpenRouter API. Check your internet connection.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error with OpenRouter API: {e}")
        raise Exception("Request timed out. Try with a smaller dataset.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception with OpenRouter API: {e}")
        if "401" in str(e):
            raise Exception("Invalid API key. Please check your OpenRouter API key.")
        elif "429" in str(e):
            raise Exception("Rate limit exceeded. Please wait and try again.")
        else:
            raise Exception(f"OpenRouter API request failed: {e}")
    except KeyError as e:
        logger.error(f"KeyError in OpenRouter response: {e}")
        raise Exception("Unexpected response format from OpenRouter API")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in OpenRouter response: {e}")
        raise Exception("Invalid response from OpenRouter API")

def call_ollama(prompt, model_name, stream=False):
    """Call Ollama API with error handling and optional streaming."""
    logger.debug(f"Calling Ollama API with model: {model_name}, stream: {stream}")
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }
        
        logger.debug(f"Sending request to Ollama with {len(prompt)} character prompt")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=3000,  # 50 minutes timeout for large datasets
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            logger.debug("Returning streaming response object")
            return response
        else:
            result = response.json()
            logger.info("Successfully received non-streaming response from Ollama")
            return result.get('response', '')
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error with Ollama API: {e}")
        raise Exception("Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error with Ollama API: {e}")
        raise Exception("Request timed out. Try with a smaller dataset or a faster model.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception with Ollama API: {e}")
        raise Exception(f"Ollama API request failed: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in Ollama response: {e}")
        raise Exception("Invalid response from Ollama API")

def call_openrouter_stream(prompt, model_name, api_key):
    """Call OpenRouter API with streaming and return complete response."""
    logger.debug(f"Starting OpenRouter streaming with model: {model_name}")
    try:
        response = call_openrouter(prompt, model_name, api_key, stream=True)
        
        complete_response = ""
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    line_data = line.decode('utf-8')
                    if line_data.startswith('data: '):
                        line_data = line_data[6:]  # Remove 'data: ' prefix
                    
                    if line_data.strip() == '[DONE]':
                        logger.debug("Received [DONE] signal from OpenRouter stream")
                        break
                        
                    chunk = json.loads(line_data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            complete_response += content
                            chunk_count += 1
                            yield content  # Yield each chunk for streaming
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed chunk in OpenRouter stream: {e}")
                    continue
        
        logger.info(f"OpenRouter streaming completed. Received {chunk_count} chunks, total response length: {len(complete_response)}")
        return complete_response
        
    except Exception as e:
        logger.error(f"OpenRouter streaming failed: {e}")
        raise Exception(f"OpenRouter streaming failed: {e}")

def call_ollama_stream(prompt, model_name):
    """Call Ollama API with streaming and return complete response."""
    logger.debug(f"Starting Ollama streaming with model: {model_name}")
    try:
        response = call_ollama(prompt, model_name, stream=True)
        
        complete_response = ""
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        complete_response += chunk['response']
                        chunk_count += 1
                        yield chunk['response']  # Yield each chunk for streaming
                    if chunk.get('done', False):
                        logger.debug("Received done signal from Ollama stream")
                        break
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping malformed chunk in Ollama stream: {e}")
                    continue
        
        logger.info(f"Ollama streaming completed. Received {chunk_count} chunks, total response length: {len(complete_response)}")
        return complete_response
        
    except Exception as e:
        logger.error(f"Ollama streaming failed: {e}")
        raise Exception(f"Streaming failed: {e}")

def get_diverse_samples(df, num_samples=3, sample_size=50):
    """Extract diverse samples from the dataset for analysis."""
    logger.info(f"Extracting {num_samples} diverse samples of {sample_size} rows each from dataset with {len(df)} rows")
    samples = []
    total_rows = len(df)
    
    if total_rows <= sample_size * num_samples:
        # If dataset is small, just return the whole thing in chunks
        logger.debug("Dataset is small, chunking entire dataset")
        chunk_size = total_rows // num_samples
        for i in range(num_samples):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_samples - 1 else total_rows
            sample = df.iloc[start_idx:end_idx].copy()
            samples.append(sample)
            logger.debug(f"Created chunk sample {i+1}: rows {start_idx}-{end_idx-1}")
    else:
        # For large datasets, get diverse samples
        logger.debug("Dataset is large, extracting diverse samples")
        
        # Sample 1: First rows (often headers and typical data)
        samples.append(df.head(sample_size).copy())
        logger.debug(f"Sample 1: First {sample_size} rows")
        
        # Sample 2: Random middle section
        start_idx = random.randint(sample_size, total_rows - sample_size * 2)
        samples.append(df.iloc[start_idx:start_idx + sample_size].copy())
        logger.debug(f"Sample 2: Random middle section, rows {start_idx}-{start_idx + sample_size - 1}")
        
        # Sample 3: Last rows (often contains edge cases)
        samples.append(df.tail(sample_size).copy())
        logger.debug(f"Sample 3: Last {sample_size} rows")
        
        # Additional samples for very large datasets
        if num_samples > 3:
            for i in range(num_samples - 3):
                start_idx = random.randint(0, total_rows - sample_size)
                samples.append(df.iloc[start_idx:start_idx + sample_size].copy())
                logger.debug(f"Additional sample {i+4}: rows {start_idx}-{start_idx + sample_size - 1}")
    
    logger.info(f"Successfully extracted {len(samples)} samples")
    return samples

def analyze_combined_samples(samples, provider, model_name, api_key=None, stream_container=None, enable_streaming=False):
    """Analyze all samples together to identify comprehensive data quality issues."""
    logger.info(f"Starting combined analysis of {len(samples)} samples using {provider}/{model_name}")
    
    # Combine all samples into a single continuous dataset
    combined_df = pd.concat(samples, ignore_index=True)
    combined_data = combined_df.to_csv(index=False)
    
    total_rows = len(combined_df)
    logger.debug(f"Combined {len(samples)} samples into single dataset: {total_rows} total rows")
    logger.debug(f"Combined dataset shape: {combined_df.shape}")
    
    prompt = f"""
You are a data quality analyst. Analyze this dataset (combined from {len(samples)} diverse samples) to identify comprehensive data quality issues and provide structured recommendations:

{combined_data}

Provide a detailed analysis of data quality issues in this format:

COMPREHENSIVE_ISSUES_FOUND:
- List each specific issue found in the dataset
- Include column names and example problematic values
- Categorize issues (missing values, format inconsistencies, invalid data, duplicates, etc.)
- Note frequency and distribution of each issue

AFFECTED_COLUMNS:
- List ALL columns that have issues
- Specify the type of issue for each column
- Include severity level for each column's issues
- For each column, specify the data type (text, numeric, email, date, categorical, etc.)

COLUMN_ANALYSIS:
For each problematic column, provide:
- Column Name: [name]
- Data Type: [inferred type: text/numeric/email/date/phone/categorical/etc.]
- Issues Found: [specific issues]
- Missing Value %: [percentage]
- Invalid Data Examples: [examples if any]
- Recommended Strategies: [list 2-3 appropriate strategies for this specific column type]

DATA_QUALITY_PATTERNS:
- Identify overall data quality patterns
- Note any systematic issues or anomalies
- Highlight the most problematic areas of the dataset

SEVERITY_ASSESSMENT:
- Rate the overall data quality: HIGH/MEDIUM/LOW
- Explain the most critical issues that need fixing
- Prioritize issues by impact and frequency

GLOBAL_STRATEGY_OPTIONS:
- Simple approach 1: [describe a conservative global strategy]
- Simple approach 2: [describe a balanced global strategy]
- Simple approach 3: [describe an aggressive global strategy]

Be comprehensive and detailed. Focus on actionable issues that can be programmatically fixed.
"""
    
    try:
        if provider == "OpenRouter":
            if enable_streaming and stream_container:
                logger.debug("Using OpenRouter streaming for combined sample analysis")
                response = ""
                for chunk in call_openrouter_stream(prompt, model_name, api_key):
                    response += chunk
                    with stream_container.container():
                        st.markdown(f"**🔍 Analyzing Combined Dataset ({total_rows} rows from {len(samples)} samples)...**")
                        st.text(response[-500:])
                
            else:
                logger.debug("Using OpenRouter non-streaming for combined sample analysis")
                response = call_openrouter(prompt, model_name, api_key, stream=False)
        else:
            if enable_streaming and stream_container:
                logger.debug("Using Ollama streaming for combined sample analysis")
                response = ""
                for chunk in call_ollama_stream(prompt, model_name):
                    response += chunk
                    with stream_container.container():
                        st.markdown(f"**🔍 Analyzing Combined Dataset ({total_rows} rows from {len(samples)} samples)...**")
                        st.text(response[-500:])
            else:
                logger.debug("Using Ollama non-streaming for combined sample analysis")
                response = call_ollama(prompt, model_name, stream=False)
        
        logger.info(f"Completed combined analysis, response length: {len(response)} characters")
        
        logger.debug(f"Prompt used for combined analysis:\n{prompt}\n")

        # Log the LLM response for debugging
        logger.info("=== COMBINED ANALYSIS LLM RESPONSE ===")
        logger.info(response)
        logger.info("=== END COMBINED ANALYSIS RESPONSE ===")
        
        return response
    except Exception as e:
        logger.error(f"Failed to analyze combined samples: {e}")
        raise Exception(f"Failed to analyze combined samples: {e}")

def parse_analysis_for_strategies(analysis_text):
    """Parse the LLM analysis to extract column information and strategy options."""
    logger.debug("Parsing analysis text for strategy extraction")
    
    columns_info = {}
    global_strategies = []
    
    try:
        # Extract column analysis section
        if "COLUMN_ANALYSIS:" in analysis_text:
            column_section = analysis_text.split("COLUMN_ANALYSIS:")[1].split("DATA_QUALITY_PATTERNS:")[0]
            
            # Parse individual column entries
            column_entries = []
            current_entry = ""
            
            for line in column_section.split('\n'):
                line = line.strip()
                # Handle various markdown patterns for column name detection
                clean_line = line.replace("*", "").replace("-", "").strip()
                if ("Column Name:" in clean_line):
                    if current_entry:
                        column_entries.append(current_entry)
                    current_entry = line
                elif current_entry and line:
                    current_entry += "\n" + line
            
            if current_entry:
                column_entries.append(current_entry)
            
            # Parse each column entry
            for entry in column_entries:
                try:
                    lines = entry.split('\n')
                    col_name = None
                    col_type = "unknown"
                    issues = []
                    missing_pct = 0
                    strategies = []
                    
                    for line in lines:
                        # Clean line from markdown formatting
                        clean_line = line.strip().replace("*", "").replace("-", "").strip()
                        
                        if "Column Name:" in clean_line:
                            # Extract column name and remove markdown backticks
                            col_name = clean_line.split(":", 1)[1].strip()
                            # Remove markdown backticks if present
                            col_name = col_name.strip('`')
                            logger.debug(f"Extracted column name: '{col_name}' from line: '{line}'")
                        elif "Data Type:" in clean_line:
                            col_type = clean_line.split(":", 1)[1].strip().lower()
                        elif "Issues Found:" in clean_line:
                            issues = [clean_line.split(":", 1)[1].strip()]
                        elif "Missing Value %" in clean_line:
                            try:
                                pct_text = clean_line.split(":", 1)[1].strip()
                                # Handle "Approximately X%" format
                                pct_text = pct_text.replace("Approximately", "").replace("%", "").strip()
                                missing_pct = float(pct_text)
                            except:
                                missing_pct = 0
                        elif "Recommended Strategies:" in clean_line:
                            strategies = [s.strip() for s in clean_line.split(":", 1)[1].split(",")]
                    
                    if col_name:
                        columns_info[col_name] = {
                            'type': col_type,
                            'issues': issues,
                            'missing_percentage': missing_pct,
                            'recommended_strategies': strategies
                        }
                        logger.debug(f"Added column info for '{col_name}': {columns_info[col_name]}")
                except Exception as e:
                    logger.warning(f"Error parsing column entry: {e}")
                    continue
        
        # Extract global strategies
        if "GLOBAL_STRATEGY_OPTIONS:" in analysis_text:
            global_section = analysis_text.split("GLOBAL_STRATEGY_OPTIONS:")[1]
            for line in global_section.split('\n'):
                line = line.strip()
                if line.startswith("- Simple approach"):
                    global_strategies.append(line.replace("- ", ""))
        
        logger.info(f"Parsed {len(columns_info)} columns and {len(global_strategies)} global strategies")
        return columns_info, global_strategies
        
    except Exception as e:
        logger.error(f"Error parsing analysis: {e}")
        return {}, []

def get_column_strategies(column_name, column_type, issues, missing_pct):
    """Get appropriate strategies for a specific column based on its type and issues."""
    
    strategies = []
    
    # Determine column type more precisely
    col_type_lower = column_type.lower()
    col_name_lower = column_name.lower()
    
    # Identify specific column types
    if any(keyword in col_name_lower for keyword in ['email', 'mail']):
        actual_type = 'email'
    elif any(keyword in col_name_lower for keyword in ['phone', 'tel', 'mobile']):
        actual_type = 'phone'
    elif any(keyword in col_name_lower for keyword in ['date', 'time', 'created', 'updated']):
        actual_type = 'date'
    elif any(keyword in col_name_lower for keyword in ['id', 'key', 'identifier']):
        actual_type = 'identifier'
    elif 'numeric' in col_type_lower or 'int' in col_type_lower or 'float' in col_type_lower:
        actual_type = 'numeric'
    elif 'categorical' in col_type_lower or 'category' in col_type_lower:
        actual_type = 'categorical'
    else:
        actual_type = 'text'
    
    # Generate strategies based on type and missing percentage
    if missing_pct > 0:
        if actual_type == 'email':
            strategies.extend([
                "Remove rows with missing emails (if email is required)",
                "Keep missing emails as null (if email is optional)",
                "Mark missing emails with placeholder (e.g., 'no-email@unknown.com')"
            ])
        elif actual_type == 'phone':
            strategies.extend([
                "Remove rows with missing phone numbers",
                "Keep missing phones as null",
                "Mark missing phones with placeholder (e.g., 'XXX-XXX-XXXX')"
            ])
        elif actual_type == 'date':
            strategies.extend([
                "Remove rows with missing dates",
                "Impute with median date",
                "Impute with current date",
                "Keep as null and handle in analysis"
            ])
        elif actual_type == 'numeric':
            strategies.extend([
                "Remove rows with missing values",
                "Impute with mean",
                "Impute with median",
                "Impute with mode",
                "Forward fill",
                "Backward fill"
            ])
        elif actual_type == 'categorical':
            strategies.extend([
                "Remove rows with missing categories",
                "Impute with mode (most frequent)",
                "Create 'Unknown' category",
                "Forward fill",
                "Backward fill"
            ])
        elif actual_type == 'identifier':
            strategies.extend([
                "Remove rows with missing IDs",
                "Generate new unique IDs",
                "Keep as null (if not primary key)"
            ])
        else:  # text
            strategies.extend([
                "Remove rows with missing text",
                "Impute with mode (most frequent)",
                "Replace with 'Unknown' or 'Not Provided'",
                "Keep as null"
            ])
    
    # Add format/validation strategies
    if actual_type == 'email':
        strategies.extend([
            "Validate email format and remove invalid",
            "Standardize email format (lowercase)",
            "Remove duplicate emails"
        ])
    elif actual_type == 'phone':
        strategies.extend([
            "Standardize phone format (e.g., +1-XXX-XXX-XXXX)",
            "Remove non-numeric characters",
            "Validate phone number length"
        ])
    elif actual_type == 'date':
        strategies.extend([
            "Standardize date format (YYYY-MM-DD)",
            "Remove future dates (if not allowed)",
            "Remove dates before reasonable threshold"
        ])
    elif actual_type == 'numeric':
        strategies.extend([
            "Remove outliers (IQR method)",
            "Cap outliers at percentiles (e.g., 95th)",
            "Standardize numeric format"
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_strategies = []
    for strategy in strategies:
        if strategy not in seen:
            seen.add(strategy)
            unique_strategies.append(strategy)
    
    return unique_strategies[:8]  # Limit to 8 options to avoid overwhelming UI

def create_strategy_selection_ui(columns_info, global_strategies, combined_df):
    """Create the strategy selection interface."""
    logger.info("Creating strategy selection UI")
    
    st.subheader("🎯 Data Cleaning Strategy Selection")
    st.markdown("Choose how to handle the identified data quality issues:")
    
    # Strategy selection mode
    strategy_mode = st.radio(
        "Select Strategy Mode:",
        ["Simple (Global Rules)", "Advanced (Column-wise Control)"],
        help="Simple: Apply same rules to all columns. Advanced: Customize for each column."
    )
    
    selected_strategies = {}
    
    if strategy_mode == "Simple (Global Rules)":
        st.markdown("### 🌐 Global Strategy Selection")
        st.info("These strategies will be applied to all applicable columns.")
        
        # Missing values strategy
        missing_strategy = st.selectbox(
            "Missing Values Strategy:",
            [
                "Conservative: Remove rows with any missing critical data",
                "Balanced: Impute numerical with median, categorical with mode, remove others",
                "Aggressive: Impute all possible values, remove only when impossible"
            ],
            help="How to handle missing values across all columns"
        )
        
        # Data validation strategy  
        validation_strategy = st.selectbox(
            "Data Validation Strategy:",
            [
                "Strict: Remove all invalid data",
                "Moderate: Fix what's possible, remove what's not",
                "Permissive: Keep most data, only remove severely invalid entries"
            ],
            help="How strictly to validate and clean data formats"
        )
        
        # Outlier handling
        outlier_strategy = st.selectbox(
            "Outlier Handling:",
            [
                "Remove outliers beyond 3 standard deviations",
                "Cap outliers at 95th/5th percentiles", 
                "Keep all outliers",
                "Remove outliers beyond IQR * 1.5"
            ],
            help="How to handle statistical outliers in numeric data"
        )
        
        # Store global strategies
        selected_strategies['global'] = {
            'missing': missing_strategy,
            'validation': validation_strategy,
            'outliers': outlier_strategy
        }
        
    else:  # Advanced mode
        st.markdown("### 🔧 Advanced Column-wise Strategy Selection")
        st.info("Customize cleaning strategies for each problematic column.")
        
        if not columns_info:
            st.warning("No column-specific information available. Using simple mode.")
            return {'global': {'missing': 'balanced', 'validation': 'moderate', 'outliers': 'iqr'}}
        
        # Create tabs for different issue types
        tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Validation", "Format Issues"])
        
        with tab1:
            st.markdown("#### Missing Values Strategy by Column")
            missing_strategies = {}
            
            for col_name, col_info in columns_info.items():
                if col_info['missing_percentage'] > 0:
                    col_strategies = get_column_strategies(
                        col_name, col_info['type'], col_info['issues'], col_info['missing_percentage']
                    )
                    
                    # Filter for missing value strategies
                    missing_options = [s for s in col_strategies if any(keyword in s.lower() for keyword in ['missing', 'remove', 'impute', 'null'])]
                    
                    if missing_options:
                        st.markdown(f"**{col_name}** ({col_info['type']}) - {col_info['missing_percentage']:.1f}% missing:")
                        selected = st.selectbox(
                            f"Strategy for {col_name}:",
                            missing_options,
                            key=f"missing_{col_name}",
                            help=f"Column type: {col_info['type']}. Issues: {', '.join(col_info['issues'])}"
                        )
                        missing_strategies[col_name] = selected
            
            selected_strategies['missing'] = missing_strategies
        
        with tab2:
            st.markdown("#### Data Validation Strategy by Column")
            validation_strategies = {}
            
            for col_name, col_info in columns_info.items():
                col_strategies = get_column_strategies(
                    col_name, col_info['type'], col_info['issues'], col_info['missing_percentage']
                )
                
                # Filter for validation strategies
                validation_options = [s for s in col_strategies if any(keyword in s.lower() for keyword in ['validate', 'format', 'standardize', 'remove invalid'])]
                
                if validation_options:
                    st.markdown(f"**{col_name}** ({col_info['type']}):")
                    selected = st.selectbox(
                        f"Validation for {col_name}:",
                        validation_options,
                        key=f"validation_{col_name}",
                        help=f"Issues found: {', '.join(col_info['issues'])}"
                    )
                    validation_strategies[col_name] = selected
            
            selected_strategies['validation'] = validation_strategies
        
        with tab3:
            st.markdown("#### Format Standardization by Column")
            format_strategies = {}
            
            for col_name, col_info in columns_info.items():
                col_strategies = get_column_strategies(
                    col_name, col_info['type'], col_info['issues'], col_info['missing_percentage']
                )
                
                # Filter for format strategies
                format_options = [s for s in col_strategies if any(keyword in s.lower() for keyword in ['format', 'standard', 'lowercase', 'remove non'])]
                
                if format_options:
                    st.markdown(f"**{col_name}** ({col_info['type']}):")
                    selected = st.selectbox(
                        f"Format for {col_name}:",
                        format_options,
                        key=f"format_{col_name}",
                        help=f"Current format issues in this column"
                    )
                    format_strategies[col_name] = selected
            
            selected_strategies['format'] = format_strategies
    
    # Preview selected strategies
    with st.expander("📋 Selected Strategies Summary", expanded=False):
        st.json(selected_strategies)
    
    # Validation warnings
    validation_warnings = validate_strategy_selection(selected_strategies, columns_info)
    if validation_warnings:
        st.warning("⚠️ Strategy Validation Warnings:")
        for warning in validation_warnings:
            st.write(f"- {warning}")
    
    return selected_strategies

def validate_strategy_selection(selected_strategies, columns_info):
    """Validate that selected strategies are appropriate for column types."""
    warnings = []
    
    try:
        if 'missing' in selected_strategies:
            for col_name, strategy in selected_strategies['missing'].items():
                if col_name in columns_info:
                    col_type = columns_info[col_name]['type'].lower()
                    
                    # Check for inappropriate imputation
                    if 'impute' in strategy.lower():
                        if any(keyword in col_type for keyword in ['email', 'phone', 'identifier', 'id']):
                            warnings.append(f"'{col_name}': Imputation may not be appropriate for {col_type} columns")
                        
                        if any(keyword in col_name.lower() for keyword in ['email', 'phone', 'id']):
                            warnings.append(f"'{col_name}': Consider removal instead of imputation for this column type")
                    
                    # Check for overly aggressive removal
                    missing_pct = columns_info[col_name].get('missing_percentage', 0)
                    if 'remove' in strategy.lower() and missing_pct > 50:
                        warnings.append(f"'{col_name}': Removing {missing_pct:.1f}% of data may be too aggressive")
        
        if 'validation' in selected_strategies:
            for col_name, strategy in selected_strategies['validation'].items():
                if col_name in columns_info:
                    col_type = columns_info[col_name]['type'].lower()
                    
                    # Check format validation appropriateness
                    if 'email' in col_type and 'email' not in strategy.lower():
                        warnings.append(f"'{col_name}': Consider email-specific validation for email columns")
                    
                    if 'phone' in col_type and 'phone' not in strategy.lower():
                        warnings.append(f"'{col_name}': Consider phone-specific validation for phone columns")
    
    except Exception as e:
        logger.warning(f"Error validating strategies: {e}")
    
    return warnings

def generate_cleaning_code(combined_analysis, df_info, provider, model_name, api_key=None, stream_container=None, enable_streaming=False, selected_strategies=None):
    """Generate Python code to clean the entire dataset based on combined sample analysis and user strategies."""
    logger.info(f"Generating cleaning code using {provider}/{model_name} based on combined sample analysis")
    
    # Create a summary of the dataset structure
    df_structure = f"""
Dataset Structure:
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Data Types: {df_info['dtypes']}
- Memory Usage: {df_info['memory_usage']} MB
"""
    
    # Include user strategies if provided
    strategy_section = ""
    if selected_strategies:
        strategy_section = f"""

USER SELECTED STRATEGIES:
{selected_strategies}

IMPORTANT: You MUST implement the cleaning code according to the user's selected strategies above. 
For each column and issue type, use the exact strategy the user chose.
"""
    
    prompt = f"""
You are a data cleaning expert. Based on the following comprehensive analysis of multiple samples from a dataset, generate complete Python code to clean the entire dataset.

{df_structure}

{combined_analysis}

{strategy_section}

IMPORTANT: You MUST generate a complete Python function that follows this EXACT structure:

CODE:
```python
def clean_dataset(df):
    \"\"\"
    Clean the dataset based on comprehensive analysis of multiple samples and user-selected strategies.
    Returns: cleaned DataFrame
    \"\"\"
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    
    print("Starting comprehensive data cleaning process...")
    original_shape = df.shape
    print(f"Original dataset shape: {{original_shape}}")
    
    # Step 1: Handle missing values (implement user-selected strategies)
    # [Your cleaning code here based on the analysis and user strategies]
    
    # Step 2: Fix data types
    # [Your cleaning code here]
    
    # Step 3: Standardize formats (implement user-selected validation strategies)
    # [Your cleaning code here]
    
    # Step 4: Remove invalid data
    # [Your cleaning code here]
    
    # Step 5: Final validation
    # [Your cleaning code here]
    
    print(f"Cleaning completed. Final shape: {{df.shape}}")
    return df
```

REQUIREMENTS:
1. The function MUST be named exactly "clean_dataset"
2. It MUST take a pandas DataFrame as the only parameter
3. It MUST return the cleaned DataFrame
4. Include print statements for logging each major step
5. Handle all issues identified in the analysis above
6. Apply operations in the correct order to avoid conflicts
7. Use proper error handling for edge cases
8. IMPLEMENT THE USER'S SELECTED STRATEGIES EXACTLY as specified above

Generate ONLY the function code inside the CODE block. Do not include any explanations or additional text outside the code block.

EXPLANATION:
[After the code, provide a detailed explanation of what the code does and how it addresses the issues found in the analysis and implements the user's strategies]

Make the code robust and comprehensive. Address all issues identified in the combined analysis according to user preferences.

"""
    
    try:
        if provider == "OpenRouter":
            if enable_streaming and stream_container:
                logger.debug("Using OpenRouter streaming for code generation")
                response = ""
                for chunk in call_openrouter_stream(prompt, model_name, api_key):
                    response += chunk
                    with stream_container.container():
                        st.markdown("**🔧 Generating Comprehensive Cleaning Code...**")
                        st.text(response[-500:])
            else:
                logger.debug("Using OpenRouter non-streaming for code generation")
                response = call_openrouter(prompt, model_name, api_key, stream=False)
        else:
            if enable_streaming and stream_container:
                logger.debug("Using Ollama streaming for code generation")
                response = ""
                for chunk in call_ollama_stream(prompt, model_name):
                    response += chunk
                    with stream_container.container():
                        st.markdown("**🔧 Generating Comprehensive Cleaning Code...**")
                        st.text(response[-500:])
            else:
                logger.debug("Using Ollama non-streaming for code generation")
                response = call_ollama(prompt, model_name, stream=False)
        
        logger.info(f"Generated cleaning code, response length: {len(response)} characters")
        logger.debug(f"Prompt used for code generation:\n{prompt}\n")

        # Log the LLM response for debugging
        logger.info("=== CODE GENERATION LLM RESPONSE ===")
        logger.info(response)
        logger.info("=== END CODE GENERATION RESPONSE ===")
        
        return response
    except Exception as e:
        logger.error(f"Failed to generate cleaning code: {e}")
        raise Exception(f"Failed to generate cleaning code: {e}")

def extract_code_from_response(response):
    """Extract Python code from LLM response."""
    logger.debug(f"Extracting code from response of length {len(response)}")
    
    # Log the full response for debugging
    logger.debug("=== FULL LLM RESPONSE FOR CODE EXTRACTION ===")
    logger.debug(response)
    logger.debug("=== END FULL RESPONSE ===")
    
    # Method 1: Look for code blocks with ```python
    if "```python" in response:
        logger.debug("Found python code block markers (Method 1)")
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1:
            code = response[start:end].strip()
            logger.info(f"Extracted code block: {len(code)} characters, {len(code.split(chr(10)))} lines")
            if "def clean_dataset" in code:
                logger.info(f"Found clean_dataset function in extracted code. \n{code}")
                return code
            else:
                logger.warning("No clean_dataset function found in extracted code block")
    
    # Method 2: Look for CODE: section
    elif "CODE:" in response:
        logger.debug("Found CODE: marker (Method 2)")
        start = response.find("CODE:") + 5
        if "```python" in response[start:]:
            code_start = response.find("```python", start) + 9
            code_end = response.find("```", code_start)
            if code_end != -1:
                code = response[code_start:code_end].strip()
                logger.info(f"Extracted code from CODE section: {len(code)} characters")
                if "def clean_dataset" in code:
                    logger.info(f"Found clean_dataset function in CODE section. \n{code}")
                    return code
                else:
                    logger.warning("No clean_dataset function found in CODE section")
        else:
            # Try to extract everything after CODE: until next section
            lines = response[start:].split('\n')
            code_lines = []
            for line in lines:
                if line.strip().startswith('EXPLANATION:') or line.strip().startswith('###'):
                    break
                code_lines.append(line)
            code = '\n'.join(code_lines).strip()
            if code and "def clean_dataset" in code:
                logger.info(f"Extracted code from CODE section (no markers): {len(code)} characters. \n{code}")
                return code
    
    # Method 3: Look for any function definition
    if "def clean_dataset" in response:
        logger.debug("Attempting to extract function definition (Method 3)")
        start = response.find("def clean_dataset")
        # Find the end of the function (next function or end of response)
        lines = response[start:].split('\n')
        function_lines = []
        indent_level = None
        in_function = False
        
        for line in lines:
            if line.strip() == "":
                function_lines.append(line)
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            if line.strip().startswith("def clean_dataset"):
                function_lines.append(line)
                indent_level = current_indent
                in_function = True
            elif in_function:
                # Continue if indented more than function definition or empty line
                if current_indent > indent_level or line.strip() == "":
                    function_lines.append(line)
                elif line.strip().startswith('```') or line.strip().startswith('EXPLANATION:'):
                    # Stop at code block end or explanation section
                    break
                elif current_indent <= indent_level and line.strip() and not line.startswith(' '):
                    # Stop at same or less indentation (next function or section)
                    break
                else:
                    function_lines.append(line)
        
        code = '\n'.join(function_lines).strip()
        if code:
            logger.info(f"Extracted function definition: {len(code)} characters, {len(function_lines)} lines. \n{code}")
            return code
    
    # Method 4: Look for any Python-like code with imports and function
    if any(keyword in response for keyword in ["import pandas", "import numpy", "def ", "pd.", "np."]):
        logger.debug("Found Python-like code, attempting to extract it (Method 4)")
        lines = response.split('\n')
        code_lines = []
        found_code = False
        
        for line in lines:
            # Start collecting when we see imports or function definitions
            if any(keyword in line for keyword in ["import pandas", "import numpy", "def clean", "def process"]):
                found_code = True
            
            if found_code:
                # Stop at explanation sections or markdown
                if line.strip().startswith('EXPLANATION:') or line.strip().startswith('###') or line.strip().startswith('**'):
                    break
                code_lines.append(line)
        
        code = '\n'.join(code_lines).strip()
        if code and len(code) > 100:  # Ensure we have substantial code
            logger.info(f"Extracted Python-like code: {len(code)} characters. \n{code}")
            
            # If no clean_dataset function, try to wrap the code
            if "def clean_dataset" not in code:
                logger.info("No clean_dataset function found, attempting to wrap code")
                wrapped_code = f"""def clean_dataset(df):
    \"\"\"
    Clean the dataset based on analysis.
    Returns: cleaned DataFrame
    \"\"\"
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    
    print("Starting data cleaning process...")
    original_shape = df.shape
    print(f"Original dataset shape: {{original_shape}}")
    
{chr(10).join('    ' + line if line.strip() else line for line in code.split(chr(10)))}
    
    print(f"Cleaning completed. Final shape: {{df.shape}}")
    return df
"""
                logger.info(f"Created wrapped clean_dataset function. \n{wrapped_code}")
                return wrapped_code
            else:
                return code
    
    logger.error("Could not extract Python code from response - no recognizable patterns found")
    logger.error("Response analysis:")
    logger.error(f"- Contains 'def clean_dataset': {'def clean_dataset' in response}")
    logger.error(f"- Contains '```python': {'```python' in response}")
    logger.error(f"- Contains 'CODE:': {'CODE:' in response}")
    logger.error(f"- Contains 'import pandas': {'import pandas' in response}")
    logger.error(f"- Response length: {len(response)} characters")
    
    raise ValueError("Could not extract Python code from response")

def execute_cleaning_code(df, cleaning_code, log_container):
    """Safely execute the generated cleaning code."""
    logger.info(f"Executing cleaning code on dataset with shape {df.shape}")
    logger.info(f"Dataset(head) : {df.head()}")
    logger.debug(f"Cleaning code length: {len(cleaning_code)} characters")
    
    # Log the actual code being executed
    logger.debug("=== CODE TO BE EXECUTED ===")
    logger.debug(cleaning_code)
    logger.debug("=== END CODE ===")
    
    try:
        # Create a safe execution environment
        safe_globals = {
            'pandas': pd,
            'pd': pd,
            'numpy': np,
            'np': np,
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'print': lambda *args: log_step("Code Execution", " ".join(map(str, args)), log_container)
        }
        
        # Execute the code
        logger.debug("Executing cleaning code in safe environment")
        exec(cleaning_code, safe_globals)
        
        # Log what functions are available after execution
        available_functions = [name for name, obj in safe_globals.items() if callable(obj)]
        logger.debug(f"Available functions after execution: {available_functions}")
        
        # Get the cleaning function
        clean_function = safe_globals.get('clean_dataset')
        if clean_function is None:
            logger.error("No 'clean_dataset' function found in generated code")
            logger.error("Available functions in safe_globals:")
            for name, obj in safe_globals.items():
                if callable(obj):
                    logger.error(f"  - {name}: {type(obj)}")
            raise ValueError("No 'clean_dataset' function found in generated code")
        
        # Apply the cleaning function
        logger.debug("Applying cleaning function to dataset")
        log_step("Code Execution", "Applying cleaning function to dataset...", log_container)
        cleaned_df = clean_function(df.copy())
        
        logger.info(f"Cleaning code executed successfully. Dataset shape: {df.shape} → {cleaned_df.shape}")
        return cleaned_df
    
    except Exception as e:
        logger.error(f"Error executing cleaning code: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        raise Exception(f"Error executing cleaning code: {e}")

def validate_cleaning(original_samples, cleaned_df, provider, model_name, api_key=None, stream_container=None, enable_streaming=False):
    """Validate that the cleaning was successful by comparing combined before/after datasets."""
    logger.info(f"Starting unified validation process with {len(original_samples)} samples using {provider}/{model_name}")
    
    # Get new samples from the cleaned data (same sampling approach)
    new_samples = get_diverse_samples(cleaned_df, len(original_samples))
    logger.debug(f"Generated {len(new_samples)} new samples from cleaned data for validation")
    
    # Combine original samples into unified "before" dataset
    combined_original = pd.concat(original_samples, ignore_index=True)
    original_data = combined_original.to_csv(index=False)
    
    # Combine new samples into unified "after" dataset  
    combined_cleaned = pd.concat(new_samples, ignore_index=True)
    cleaned_data = combined_cleaned.to_csv(index=False)
    
    logger.debug(f"Combined validation datasets: Original {combined_original.shape}, Cleaned {combined_cleaned.shape}")
    
    prompt = f"""
Compare these unified before and after datasets to validate data cleaning effectiveness:

ORIGINAL DATASET (before cleaning - combined from {len(original_samples)} samples):
{original_data}

CLEANED DATASET (after cleaning - combined from {len(new_samples)} samples):
{cleaned_data}

Provide comprehensive validation results in this format:

ISSUES_RESOLVED:
- List specific issues that were successfully fixed across the dataset
- Compare data quality patterns before vs after
- Quantify improvements where possible (e.g., missing values reduced, formats standardized)

REMAINING_ISSUES:
- List any issues that still exist in the cleaned dataset
- Identify any new problems introduced by the cleaning process
- Note any patterns that suggest incomplete cleaning

QUALITY_IMPROVEMENT_ASSESSMENT:
- Rate overall improvement: EXCELLENT/GOOD/FAIR/POOR
- Overall data quality now: HIGH/MEDIUM/LOW
- Specific quality metrics comparison (missing values, consistency, validity)

DATA_INTEGRITY_CHECK:
- Verify no critical data was lost inappropriately
- Check that data relationships are preserved
- Confirm data types and formats are appropriate

RECOMMENDATIONS:
- Suggest any additional cleaning steps needed
- Prioritize remaining issues by severity
- Recommend any follow-up validation steps

Be thorough and focus on the overall dataset transformation quality.
"""
    
    try:
        if provider == "OpenRouter":
            if enable_streaming and stream_container:
                logger.debug("Using OpenRouter streaming for unified validation")
                response = ""
                for chunk in call_openrouter_stream(prompt, model_name, api_key):
                    response += chunk
                    with stream_container.container():
                        st.markdown(f"**✅ Validating Combined Dataset ({len(combined_original)} vs {len(combined_cleaned)} rows)...**")
                        st.text(response[-500:])
            else:
                logger.debug("Using OpenRouter non-streaming for unified validation")
                response = call_openrouter(prompt, model_name, api_key, stream=False)
        else:
            if enable_streaming and stream_container:
                logger.debug("Using Ollama streaming for unified validation")
                response = ""
                for chunk in call_ollama_stream(prompt, model_name):
                    response += chunk
                    with stream_container.container():
                        st.markdown(f"**✅ Validating Combined Dataset ({len(combined_original)} vs {len(combined_cleaned)} rows)...**")
                        st.text(response[-500:])
            else:
                logger.debug("Using Ollama non-streaming for unified validation")
                response = call_ollama(prompt, model_name, stream=False)
        
        logger.info(f"Completed unified validation, response length: {len(response)} characters")
        
        # Log validation response for debugging
        logger.info("=== UNIFIED VALIDATION LLM RESPONSE ===")
        logger.info(response)
        logger.info("=== END UNIFIED VALIDATION RESPONSE ===")
        
        # Return single unified result instead of list
        return [response]  # Keep as list for compatibility with existing code
        
    except Exception as e:
        error_msg = f"Unified validation failed: {e}"
        logger.error(error_msg)
        return [error_msg]

def generate_eda_summary(df):
    return df.describe(include='all')

def plot_distribution(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        return px.histogram(df, x=column, title=f"Distribution of {column}")
    else:
        value_counts = df[column].value_counts().head(20).reset_index()
        value_counts.columns = ['value', 'count']
        return px.bar(value_counts, x='value', y='count', title=f"Value Counts of {column} (Top 20)")

# ---- Streamlit UI ----
st.set_page_config(page_title="Advanced Data Cleaner", layout="wide")
st.title("🧹📊 Advanced Data Cleaning & Validation Agent")
st.markdown("*Analyzes multiple samples, generates custom code, and validates results*")

# Log application startup
logger.info("Advanced Data Cleaning application started")
logger.info(f"Logging configured - Log file: advanced_data_cleaner.log")

# ---- Sidebar Configuration ----
st.sidebar.header("⚙️ Configuration")

# Provider Selection
st.sidebar.subheader("🔄 LLM Provider")
provider = st.sidebar.radio(
    "Choose LLM Provider",
    options=["OpenRouter", "Ollama"],
    help="Select between cloud-based OpenRouter or local Ollama models"
)

# Initialize session state
if 'current_provider' not in st.session_state:
    st.session_state.current_provider = provider

if 'current_model' not in st.session_state:
    st.session_state.current_model = OPENROUTER_MODELS[0] if provider == "OpenRouter" else DEFAULT_OLLAMA_MODEL

# Initialize result states
if 'combined_analysis' not in st.session_state:
    st.session_state.combined_analysis = None
if 'code_response' not in st.session_state:
    st.session_state.code_response = None
if 'cleaning_code' not in st.session_state:
    st.session_state.cleaning_code = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'samples' not in st.session_state:
    st.session_state.samples = None
if 'model_used' not in st.session_state:
    st.session_state.model_used = None
if 'provider_used' not in st.session_state:
    st.session_state.provider_used = None
if 'columns_info' not in st.session_state:
    st.session_state.columns_info = {}
if 'global_strategies' not in st.session_state:
    st.session_state.global_strategies = []
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'selected_strategies' not in st.session_state:
    st.session_state.selected_strategies = None

# Update provider in session state
st.session_state.current_provider = provider

if provider == "OpenRouter":
    # OpenRouter Configuration
    st.sidebar.subheader("🌐 OpenRouter Settings")
    
    # API Key input
    api_key_input = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        value="",
        placeholder="Enter your API key (optional - default configured)",
        help="Enter your OpenRouter API key. Leave empty to use the default configured key."
    )
    
    # Use user input or fall back to default
    effective_api_key = api_key_input.strip() if api_key_input.strip() else DEFAULT_OPENROUTER_API_KEY
    st.session_state.openrouter_api_key = effective_api_key
    
    # Streaming Configuration for OpenRouter
    enable_streaming = st.sidebar.checkbox(
        "Enable Streaming Output",
        value=True,
        help="Show LLM response as it's being generated (more interactive)"
    )
    
    # Model selection - text input with dropdown suggestions
    st.sidebar.markdown("**Model Selection:**")
    
    # Quick select dropdown for recommendations
    recommended_model = st.sidebar.selectbox(
        "📋 Quick Select (Recommendations)",
        [""] + OPENROUTER_MODELS,  # Empty option first
        help="Select from recommended models or use custom input below"
    )
    
    # Custom model input
    if recommended_model:
        default_model = recommended_model
    else:
        default_model = st.session_state.current_model if st.session_state.current_model else OPENROUTER_MODELS[0]
    
    selected_model = st.sidebar.text_input(
        "🔧 Model ID (Custom/Selected)",
        value=default_model,
        help="Enter any OpenRouter model ID or select from recommendations above"
    )
    
    # Update model if dropdown selection changed
    if recommended_model and recommended_model != selected_model:
        selected_model = recommended_model
    
    st.session_state.current_model = selected_model

else:
    # Ollama Configuration
    st.sidebar.subheader("🏠 Ollama Settings")
    
    ollama_url = st.sidebar.text_input(
        "Ollama Base URL",
        value=OLLAMA_BASE_URL,
        help="URL where Ollama is running (default: http://localhost:11434)"
    )
    
    # Update global URL if changed
    if ollama_url != OLLAMA_BASE_URL:
        OLLAMA_BASE_URL = ollama_url
    
    # Streaming Configuration
    enable_streaming = st.sidebar.checkbox(
        "Enable Streaming Output",
        value=True,
        help="Show LLM response as it's being generated (more interactive)"
    )
    
    # Model Selection
    if st.sidebar.button("🔄 Refresh Models", help="Refresh the list of available models"):
        st.rerun()

    available_models = get_available_ollama_models()

    if available_models:
        # Model selection dropdown
        if st.session_state.current_model in available_models:
            default_index = available_models.index(st.session_state.current_model)
        else:
            default_index = 0

        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=default_index,
            help="Choose from your installed Ollama models"
        )
        
        st.session_state.current_model = selected_model
        st.sidebar.success(f"✅ {len(available_models)} models available")
        
    else:
        # Manual model input if can't get model list
        st.sidebar.warning("⚠️ Could not connect to Ollama or no models found")
        
        selected_model = st.sidebar.text_input(
            "Model Name",
            value=st.session_state.current_model,
            help="Enter the name of an Ollama model (e.g., llama3.2, mistral, codellama)"
        )
        
        st.session_state.current_model = selected_model

# Advanced Settings
st.sidebar.subheader("🔧 Advanced Settings")
num_samples = st.sidebar.slider(
    "Number of Samples to Analyze",
    min_value=2,
    max_value=5,
    value=3,
    help="More samples = better analysis but slower processing"
)

sample_size = st.sidebar.slider(
    "Sample Size (rows)",
    min_value=10,
    max_value=100,
    value=50,
    help="Rows per sample for analysis"
)

# Display current configuration
st.sidebar.markdown("---")
st.sidebar.info(f"**Provider:** {provider}")
st.sidebar.info(f"**Model:** {st.session_state.current_model}")

if provider == "OpenRouter":
    st.sidebar.info(f"**API Key:** {'✅ Set' if effective_api_key else '❌ Missing'}")
    if 'enable_streaming' in locals():
        st.sidebar.info(f"**Streaming:** {'✅ Enabled' if enable_streaming else '❌ Disabled'}")
else:
    st.sidebar.info(f"**Ollama URL:** {OLLAMA_BASE_URL}")
    if 'enable_streaming' in locals():
        st.sidebar.info(f"**Streaming:** {'✅ Enabled' if enable_streaming else '❌ Disabled'}")

# Debug section to show current session state
if st.sidebar.checkbox("🔧 Show Debug Info", help="Show current session state for troubleshooting"):
    st.sidebar.write("**Session State:**")
    st.sidebar.write(f"cleaned_df: {'✅' if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None else '❌'}")
    st.sidebar.write(f"combined_analysis: {'✅' if 'combined_analysis' in st.session_state and st.session_state.combined_analysis else '❌'}")
    st.sidebar.write(f"cleaning_code: {'✅' if 'cleaning_code' in st.session_state and st.session_state.cleaning_code else '❌'}")
    st.sidebar.write(f"validation_results: {'✅' if 'validation_results' in st.session_state and st.session_state.validation_results else '❌'}")
    if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
        st.sidebar.success("Results should be visible below!")

# Prominent results indicator
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.success("✅ **RESULTS READY!**")
    st.sidebar.info("📊 Scroll down to see detailed results and analysis")
    if st.sidebar.button("🔄 Refresh Results", help="Force refresh the results display"):
        st.rerun()

# Main UI
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

# Check if results are available and show prominently
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    st.success("🎉 **Data cleaning results are ready!** Scroll down to view the complete analysis.")
    if st.button("📊 Jump to Results", help="Scroll down to see the detailed results"):
        st.write("👇 **Results are displayed below in the 'Results & Analysis' section**")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    logger.info(f"Loaded CSV file: {uploaded_file.name}, shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Dataset overview
    st.subheader("📄 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    logger.debug(f"Dataset columns: {list(df.columns)}")
    logger.debug(f"Data types: {df.dtypes.to_dict()}")
    
    # Show preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    # Check if analysis has been completed
    if not st.session_state.analysis_completed:
        if st.button("� Analyze Data Quality"):
            logger.info("Starting data quality analysis")
            logger.info(f"Configuration: Provider={provider}, Model={st.session_state.current_model}, Samples={num_samples}, Sample_size={sample_size}, Streaming={enable_streaming}")
            
            if provider == "OpenRouter" and not effective_api_key:
                logger.error("No OpenRouter API key available")
                st.error("No API key available. Please enter your OpenRouter API key or check the environment configuration.")
            else:
                # Create containers for the process
                progress_bar = st.progress(0)
                status_container = st.empty()
                log_container = st.container()
                stream_container = st.empty()
                
                try:
                    # Initialize progress
                    total_steps = 2  # Extract samples + Analyze
                    current_step = 0
                    
                    # Step 1: Extract samples
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    status_container.info(f"📊 Step {current_step}/{total_steps}: Extracting diverse samples from dataset...")
                    logger.info(f"Step {current_step}/{total_steps}: Starting sample extraction")
                    
                    samples = get_diverse_samples(df, num_samples, sample_size)
                    log_step("Sample Extraction", f"Extracted {len(samples)} samples of {sample_size} rows each", log_container)
                    
                    # Store dataset info for code generation
                    df_info = {
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict(),
                        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
                    }
                    logger.debug(f"Dataset info prepared: {df_info}")
                    
                    # Step 2: Analyze samples
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    status_container.info(f"🔍 Step {current_step}/{total_steps}: Analyzing combined samples for data quality issues...")
                    logger.info(f"Step {current_step}/{total_steps}: Starting combined sample analysis")
                    
                    log_step("Combined Analysis", f"Analyzing all {len(samples)} samples together for comprehensive insights", log_container)
                    combined_analysis = analyze_combined_samples(
                        samples, provider, st.session_state.current_model, 
                        effective_api_key if provider == "OpenRouter" else None,
                        stream_container, enable_streaming
                    )

                    st.write("**🔍 Combined Analysis Results:**")
                    st.markdown(combined_analysis)
                    
                    logger.info(f"Completed combined analysis of all samples")
                    
                    # Parse analysis for strategy creation
                    columns_info, global_strategies = parse_analysis_for_strategies(combined_analysis)
                    
                    # Store results in session state
                    st.session_state.samples = samples
                    st.session_state.combined_analysis = combined_analysis
                    st.session_state.columns_info = columns_info
                    st.session_state.global_strategies = global_strategies
                    st.session_state.df_info = df_info
                    st.session_state.current_df = df
                    st.session_state.analysis_completed = True
                    
                    # Complete analysis
                    progress_bar.progress(1.0)
                    status_container.success("🎉 Data quality analysis completed! Now select your cleaning strategies below.")
                    stream_container.empty()
                    
                    logger.info("Analysis completed and stored in session state")
                    st.rerun()  # Refresh to show strategy selection
                    
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    st.error(f"Analysis failed: {e}")
                    progress_bar.empty()
                    status_container.empty()
                    stream_container.empty()
    
    else:
        # Analysis completed, show strategy selection
        st.success("✅ Data quality analysis completed!")
        
        # Display analysis results
        with st.expander("🔍 View Analysis Results", expanded=False):
            st.markdown(st.session_state.combined_analysis)
        
        # Strategy Selection Section
        st.markdown("---")
        st.header("🎯 Strategy Selection")
        
        # Create strategy selection UI
        combined_df_for_analysis = pd.concat(st.session_state.samples, ignore_index=True)
        selected_strategies = create_strategy_selection_ui(
            st.session_state.columns_info, 
            st.session_state.global_strategies, 
            combined_df_for_analysis
        )
        
        # Store selected strategies
        st.session_state.selected_strategies = selected_strategies
        
        # Proceed button
        if st.button("✅ Proceed with Data Cleaning", type="primary"):
            logger.info("Starting data cleaning with selected strategies")
            
            # Create containers for the cleaning process
            progress_bar = st.progress(0)
            status_container = st.empty()
            log_container = st.container()
            stream_container = st.empty()
            
            try:
                # Initialize progress for cleaning steps
                total_steps = 3  # Generate code + Execute + Validate
                current_step = 0
                
                # Step 1: Generate cleaning code
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_container.info(f"🔧 Step {current_step}/{total_steps}: Generating custom cleaning code with your strategies...")
                logger.info(f"Step {current_step}/{total_steps}: Starting code generation with user strategies")
                
                log_step("Code Generation", "Generating Python code based on comprehensive sample analysis and user-selected strategies", log_container)
                code_response = generate_cleaning_code(
                    st.session_state.combined_analysis, st.session_state.df_info, provider, st.session_state.current_model,
                    effective_api_key if provider == "OpenRouter" else None,
                    stream_container, enable_streaming, selected_strategies
                )
                
                # Extract and validate code
                cleaning_code = extract_code_from_response(code_response)
                log_step("Code Extraction", f"Extracted {len(cleaning_code.split())} lines of cleaning code", log_container)
                
                # Step 2: Execute cleaning
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_container.info(f"⚙️ Step {current_step}/{total_steps}: Executing cleaning code on full dataset...")
                logger.info(f"Step {current_step}/{total_steps}: Starting code execution")
                
                cleaned_df = execute_cleaning_code(st.session_state.current_df, cleaning_code, log_container)
                log_step("Cleaning Execution", f"Dataset cleaned: {st.session_state.current_df.shape} → {cleaned_df.shape}", log_container)
                
                # Step 3: Validate results
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_container.info(f"✅ Step {current_step}/{total_steps}: Validating cleaning results...")
                logger.info(f"Step {current_step}/{total_steps}: Starting validation")
                
                log_step("Validation", "Re-analyzing samples to validate cleaning effectiveness", log_container)
                validation_results = validate_cleaning(
                    st.session_state.samples, cleaned_df, provider, st.session_state.current_model,
                    effective_api_key if provider == "OpenRouter" else None,
                    stream_container, enable_streaming
                )
                
                # Store final results in session state
                st.session_state.original_df = st.session_state.current_df
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_code = cleaning_code
                st.session_state.code_response = code_response
                st.session_state.validation_results = validation_results
                st.session_state.model_used = st.session_state.current_model
                st.session_state.provider_used = provider
                # Store strategies if they exist
                if 'selected_strategies' in st.session_state:
                    st.session_state.strategies_used = st.session_state.selected_strategies
                
                logger.info("All results stored in session state")
                
                # Complete
                progress_bar.progress(1.0)
                status_container.success("🎉 Advanced data cleaning completed successfully!")
                stream_container.empty()
                
                log_step("Process Complete", "All steps completed successfully", log_container)
                
                # Reset analysis flag to allow new analysis
                st.session_state.analysis_completed = False
                st.rerun()  # Refresh to show results
                
            except Exception as e:
                logger.error(f"Cleaning process failed: {e}")
                st.error(f"Cleaning process failed: {e}")
                progress_bar.empty()
                status_container.empty()
                stream_container.empty()
        
        # Reset Analysis button
        if st.button("🔄 Start New Analysis", help="Reset and analyze a different dataset or with different settings"):
            # Clear analysis-related session state
            st.session_state.analysis_completed = False
            st.session_state.combined_analysis = None
            st.session_state.columns_info = {}
            st.session_state.global_strategies = []
            st.session_state.selected_strategies = None
            st.session_state.samples = None
            logger.info("Analysis state reset for new analysis")
            st.rerun()

# Display results if available
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    logger.info("Displaying results from session state")
    
    # Force debug display
    st.success("🎉 **RESULTS SECTION ACTIVATED** - Data cleaning results are available!")
    st.info(f"**Session State Check:** cleaned_df exists = {'cleaned_df' in st.session_state}, not None = {st.session_state.cleaned_df is not None if 'cleaned_df' in st.session_state else 'N/A'}")
    
    st.markdown("---")
    st.header("📈 Results & Analysis")
    
    # Debug info for troubleshooting
    with st.expander("🔧 Debug Info", expanded=False):
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**Combined Analysis Available:**", st.session_state.combined_analysis is not None)
        st.write("**Code Response Available:**", st.session_state.code_response is not None)
        st.write("**Cleaning Code Available:**", st.session_state.cleaning_code is not None)
        st.write("**Validation Results Available:**", st.session_state.validation_results is not None)
        if st.session_state.combined_analysis:
            st.write("**Analysis Length:**", len(st.session_state.combined_analysis))
        if st.session_state.code_response:
            st.write("**Code Response Length:**", len(st.session_state.code_response))
        if st.session_state.cleaning_code:
            st.write("**Cleaning Code Length:**", len(st.session_state.cleaning_code))
    
    # Results overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Before vs After")
        comparison_data = {
            'Metric': ['Rows', 'Columns', 'Missing Values', 'Memory (MB)'],
            'Original': [
                f"{len(st.session_state.original_df):,}",
                len(st.session_state.original_df.columns),
                st.session_state.original_df.isnull().sum().sum(),
                f"{st.session_state.original_df.memory_usage(deep=True).sum() / 1024**2:.1f}"
            ],
            'Cleaned': [
                f"{len(st.session_state.cleaned_df):,}",
                len(st.session_state.cleaned_df.columns),
                st.session_state.cleaned_df.isnull().sum().sum(),
                f"{st.session_state.cleaned_df.memory_usage(deep=True).sum() / 1024**2:.1f}"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        logger.debug(f"Results comparison: Original {st.session_state.original_df.shape} → Cleaned {st.session_state.cleaned_df.shape}")
    
    with col2:
        st.subheader("🤖 Processing Info")
        st.info(f"**Provider:** {st.session_state.provider_used if st.session_state.provider_used else 'Unknown'}")
        st.info(f"**Model:** {st.session_state.model_used if st.session_state.model_used else 'Unknown'}")
        st.info(f"**Samples Analyzed:** {len(st.session_state.samples) if st.session_state.samples else 0}")
        st.info(f"**Validation Checks:** {len(st.session_state.validation_results) if st.session_state.validation_results else 0}")
        
        # Show strategy information if available
        if 'strategies_used' in st.session_state and st.session_state.strategies_used:
            st.subheader("🎯 Applied Strategies")
            strategies = st.session_state.strategies_used
            if 'global' in strategies:
                st.info("**Mode:** Global Rules")
                with st.expander("View Global Strategies"):
                    for key, value in strategies['global'].items():
                        st.write(f"**{key.title()}:** {value}")
            else:
                st.info("**Mode:** Column-wise Control")
                with st.expander("View Column Strategies"):
                    for strategy_type, strategy_dict in strategies.items():
                        if strategy_dict:
                            st.write(f"**{strategy_type.title()} Strategies:**")
                            for col, strat in strategy_dict.items():
                                st.write(f"- {col}: {strat}")
        else:
            st.info("**Strategies:** Default (no custom selection)")
        
        # Add LLM output statistics
        analysis_length = len(st.session_state.combined_analysis) if st.session_state.combined_analysis else 0
        code_response_length = len(st.session_state.code_response) if st.session_state.code_response else 0
        extracted_code_length = len(st.session_state.cleaning_code) if st.session_state.cleaning_code else 0
        
        st.info(f"**Analysis Output:** {analysis_length:,} characters")
        st.info(f"**Code Response:** {code_response_length:,} characters") 
        st.info(f"**Extracted Code:** {extracted_code_length:,} characters")
        
        logger.debug(f"Processing info: {st.session_state.provider_used}/{st.session_state.model_used}, {len(st.session_state.samples) if st.session_state.samples else 0} samples")
    
    # Combined Sample Analysis
    with st.expander("🔍 Combined Sample Analysis", expanded=True):
        st.markdown("### 📊 LLM Analysis of All Samples")
        st.markdown("*This is the comprehensive analysis that was used to generate the cleaning code:*")
        
        # Show the analysis in a nice format
        if st.session_state.combined_analysis:
            st.text_area(
                "Combined Analysis Output",
                value=st.session_state.combined_analysis,
                height=400,
                help="This is the raw analysis output from the LLM after analyzing all samples"
            )
        else:
            st.warning("No combined analysis available")
    
    # Generated Code
    with st.expander("🔧 Generated Cleaning Code", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Full LLM Response")
            st.markdown("*Complete response from the LLM including explanations:*")
            if st.session_state.code_response:
                st.text_area(
                    "Full LLM Code Generation Response",
                    value=st.session_state.code_response,
                    height=300,
                    help="Complete response from the LLM when generating cleaning code"
                )
            else:
                st.warning("No code response available")
        
        with col2:
            st.markdown("### ⚙️ Extracted Executable Code")
            st.markdown("*Python function extracted and executed:*")
            if st.session_state.cleaning_code:
                st.code(st.session_state.cleaning_code, language='python')
                
                # Show code statistics
                code_lines = len(st.session_state.cleaning_code.split('\n'))
                code_chars = len(st.session_state.cleaning_code)
                st.info(f"📏 **Code Stats:** {code_lines} lines, {code_chars:,} characters")
            else:
                st.warning("No extracted code available")
    
    # Validation Results
    with st.expander("✅ Validation Results", expanded=False):
        st.markdown("### 🔍 Before vs After Sample Validation")
        st.markdown("*LLM analysis of cleaning effectiveness on sample data:*")
        
        if st.session_state.validation_results and len(st.session_state.validation_results) > 0:
            for i, validation in enumerate(st.session_state.validation_results):
                with st.container():
                    st.markdown(f"#### 📝 Validation for Sample {i+1}")
                    st.text_area(
                        f"Sample {i+1} Validation",
                        value=validation,
                        height=200,
                        key=f"validation_{i}",
                        help=f"Validation results comparing original vs cleaned data for sample {i+1}"
                    )
                    if i < len(st.session_state.validation_results) - 1:
                        st.markdown("---")
        else:
            st.warning("No validation results available")
    
    # Cleaned Data Preview
    st.subheader("✨ Cleaned Dataset")
    st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)
    
    # EDA Summary
    st.subheader("📈 EDA Summary")
    st.dataframe(generate_eda_summary(st.session_state.cleaned_df))
    
    # Visualization
    if len(st.session_state.cleaned_df.columns) > 0:
        col = st.selectbox("Select column to visualize", st.session_state.cleaned_df.columns)
        if col:
            fig = plot_distribution(st.session_state.cleaned_df, col)
            st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Download Cleaned CSV", 
            st.session_state.cleaned_df.to_csv(index=False), 
            file_name="cleaned_data.csv"
        )
        logger.debug("Cleaned CSV download button created")
    with col2:
        st.download_button(
            "📝 Download Cleaning Code", 
            st.session_state.cleaning_code, 
            file_name="cleaning_code.py"
        )
        logger.debug("Cleaning code download button created")
    with col3:
        report = f"""# Data Cleaning Report

## Dataset Information
- Original shape: {st.session_state.original_df.shape}
- Cleaned shape: {st.session_state.cleaned_df.shape}
- Model used: {st.session_state.model_used}
- Provider: {st.session_state.provider_used}

## Combined Sample Analysis
{st.session_state.combined_analysis}

## Validation Results
{chr(10).join([f"### Validation {i+1}:{chr(10)}{validation}{chr(10)}" for i, validation in enumerate(st.session_state.validation_results)])}
"""
        st.download_button(
            "📋 Download Report", 
            report, 
            file_name="cleaning_report.md"
        )
        logger.debug("Cleaning report download button created")

# Add connection status in main area
if not uploaded_file:
    logger.debug("No file uploaded, showing connection status and help")
    st.info("👆 Upload a CSV file to get started with advanced data cleaning!")
    
    # Add test data button for debugging
    if st.button("🧪 Load Test Data (for UI testing)", help="Load sample data to test the UI sections"):
        st.session_state.combined_analysis = """COMPREHENSIVE_ISSUES_FOUND:
- Missing values in 'age' column (15% of records)
- Inconsistent date formats in 'created_date' column (MM/DD/YYYY vs DD-MM-YYYY)
- Invalid email formats in 'email' column (missing @ symbol, invalid domains)
- Duplicate records based on 'user_id' field (3% of dataset)
- Text case inconsistencies in 'status' column (Active, ACTIVE, active)

AFFECTED_COLUMNS:
- age: Missing values, some negative values
- created_date: Multiple date formats, some invalid dates
- email: Format validation issues
- status: Case inconsistencies
- user_id: Duplicate values

PATTERN_ANALYSIS:
- Missing values concentrated in middle samples (data collection issues)
- Date format issues primarily in earlier samples
- Email validation problems consistent across all samples

SEVERITY_ASSESSMENT:
- Overall data quality: MEDIUM
- Most critical: Missing values and duplicates
- Impact: Medium to high for analytics

CLEANING_STRATEGY:
1. Handle missing values through imputation
2. Standardize date formats
3. Validate and clean email addresses
4. Remove duplicate records
5. Standardize text case"""
        
        st.session_state.code_response = """Here's the cleaning code based on the analysis:

CODE:
```python
def clean_dataset(df):
    \"\"\"
    Clean the dataset based on comprehensive analysis.
    Returns: cleaned DataFrame
    \"\"\"
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    
    print("Starting comprehensive data cleaning process...")
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Step 1: Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    
    # Step 2: Fix data types
    df['status'] = df['status'].str.lower()
    
    print(f"Cleaning completed. Final shape: {df.shape}")
    return df
```

EXPLANATION:
This code addresses the major data quality issues identified in the analysis."""
        
        st.session_state.cleaning_code = """def clean_dataset(df):
    \"\"\"
    Clean the dataset based on comprehensive analysis.
    Returns: cleaned DataFrame
    \"\"\"
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    
    print("Starting comprehensive data cleaning process...")
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Step 1: Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    
    # Step 2: Fix data types
    df['status'] = df['status'].str.lower()
    
    print(f"Cleaning completed. Final shape: {df.shape}")
    return df"""
        
        st.session_state.validation_results = [
            """ISSUES_RESOLVED:
- Missing values in 'age' column successfully filled with median
- Status column standardized to lowercase
- Data consistency improved

REMAINING_ISSUES:
- Some edge cases in date formatting still present
- Email validation could be more robust

QUALITY_SCORE:
- Rate improvement: GOOD
- Overall data quality now: MEDIUM-HIGH

RECOMMENDATIONS:
- Consider additional date format standardization
- Implement stricter email validation""",
            
            """ISSUES_RESOLVED:
- Duplicate records successfully removed
- Text case inconsistencies resolved
- Invalid entries handled properly

REMAINING_ISSUES:
- Minor formatting issues in some fields
- Could benefit from additional validation

QUALITY_SCORE:
- Rate improvement: EXCELLENT
- Overall data quality now: HIGH

RECOMMENDATIONS:
- Data cleaning was highly effective
- Ready for analysis"""
        ]
        
        st.session_state.model_used = "test-model"
        st.session_state.provider_used = "Test Provider"
        st.session_state.samples = [pd.DataFrame({'test': [1, 2, 3]})] * 3
        
        # Create mock dataframes
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        test_data = pd.DataFrame({
            'user_id': range(100),
            'age': np.random.randint(18, 80, 100),
            'status': np.random.choice(['active', 'ACTIVE', 'Active'], 100),
            'email': [f'user{i}@test.com' for i in range(100)]
        })
        
        st.session_state.original_df = test_data.copy()
        st.session_state.cleaned_df = test_data.copy()
        
        st.success("✅ Test data loaded! Scroll down to see the UI sections with mock data.")
        st.rerun()
    
    st.markdown("### 🚀 How Advanced Cleaning Works:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Multi-Sample Analysis:**
        - Extracts diverse samples from your dataset
        - Combines all samples for comprehensive analysis
        - Identifies patterns and variations across the entire dataset
        
        **🔧 Code Generation:**
        - Creates custom Python cleaning code
        - Based on comprehensive analysis of all samples
        - Handles edge cases and complex patterns
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Full Dataset Processing:**
        - Applies generated code to entire dataset
        - Preserves data integrity and relationships
        - Comprehensive logging of all operations
        
        **✅ Validation & Verification:**
        - Re-analyzes samples after cleaning
        - Validates that issues were resolved
        - Provides detailed before/after comparison
        """)
    
    st.markdown("""
    ### 📋 Process Overview:
    1. **Sample Extraction** - Get diverse samples from your data
    2. **Combined Analysis** - LLM analyzes all samples together for comprehensive insights  
    3. **Code Generation** - Creates custom Python cleaning code based on full analysis
    4. **Data Cleaning** - Applies code to full dataset safely
    5. **Validation** - Verifies cleaning was successful
    """)
