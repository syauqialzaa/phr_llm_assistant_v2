from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import io
from typing import Optional, Dict, Tuple
from smart_chunking import VannaMilvus
from langdetect import detect
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.llms import Ollama
from googletrans import Translator
import numpy as np
from datetime import datetime, date, time
from decimal import Decimal

load_dotenv()

# Initialize embedding function
from pymilvus import model

# Initialize Vanna-Milvus instance
vn_milvus = VannaMilvus(
    config={
        "model": "mistral-nemo",
        "embedding_function": model.DefaultEmbeddingFunction(),
        "n_results": 2,
    }
)

ollama_llm = Ollama(model="llama3.1")

# Connect to PostgreSQL
vn_milvus.connect_to_postgres(
    host=os.getenv("PG_HOST"),
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=os.getenv("PG_PORT")
)

# Train with schema
df_information_schema = vn_milvus.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
plan = vn_milvus.get_training_plan_generic(df_information_schema)
vn_milvus.train(plan=plan)

# Load and train with Excel data
def load_training_data():
    excel_file = "phr_api_training_data.xlsx"
    
    # Load DDL sheet
    ddl_df = pd.read_excel(excel_file, sheet_name='DDL')
    for _, row in ddl_df.iterrows():
        if pd.notna(row['DDL']):
            vn_milvus.train(ddl=row['DDL'])
    
    # Load Documentation sheet
    doc_df = pd.read_excel(excel_file, sheet_name='Documentation')
    for _, row in doc_df.iterrows():
        if pd.notna(row['Documentation']):
            vn_milvus.train(documentation=row['Documentation'])
    
    # Load SQL sheet
    sql_df = pd.read_excel(excel_file, sheet_name='SQL')
    for _, row in sql_df.iterrows():
        if pd.notna(row['SQL']):
            vn_milvus.train(sql=row['SQL'])
    
    # Load Question-SQL sheet
    qa_df = pd.read_excel(excel_file, sheet_name='Question-SQL')
    for _, row in qa_df.iterrows():
        if pd.notna(row['Question']) and pd.notna(row['SQL']):
            vn_milvus.train(question=row['Question'], sql=row['SQL'])

# Load training data
load_training_data()

app = FastAPI()

class Question(BaseModel):
    question: str
    timestamp: datetime

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return 'en'

def translate_to_english(text: str) -> str:
    """Translate text to English if it's not already in English"""
    try:
        detected_lang = detect_language(text)
        if detected_lang != 'en':
            translator = Translator()
            result = translator.translate(text, dest='en')
            return result.text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_visualization(df: pd.DataFrame, question: str) -> Optional[str]:
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        fig = None
        # Time series
        if len(date_cols) and len(numeric_cols):
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over time")
        # Bar chart
        elif len(categorical_cols) and len(numeric_cols):
            cat, num = categorical_cols[0], numeric_cols[0]
            grp = df.groupby(cat)[num].sum().reset_index()
            fig = px.bar(grp, x=cat, y=num, title=f"{num} by {cat}")
        # Scatter
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}")
        # Pie
        elif len(categorical_cols):
            counts = df[categorical_cols[0]].value_counts()
            fig = px.pie(names=counts.index, values=counts.values, title=f"Distribution of {categorical_cols[0]}")
        # Default: row count
        else:
            fig = go.Figure(go.Bar(x=["Rows"], y=[len(df)]))
            fig.update_layout(title='Row Count')
        # Encode
        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Visualization error: {e}")
        # Fallback simple bar
        buf = io.BytesIO()
        fallback = go.Figure(go.Bar(x=["Rows"], y=[len(df)]))
        fallback.update_layout(title='Row Count')
        fallback.write_image(buf, format='png')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_llm_visualization_explanation(df: pd.DataFrame, question: str, viz_type: str, lang: str = 'en') -> str:
    """
    Generate visualization explanation using LLM for more flexible and natural explanations
    """
    try:
        # Analyze data structure
        analysis = analyze_data_for_visualization(df)
        
        # Prepare data summary for the LLM
        data_summary = {
            'total_rows': analysis['total_rows'],
            'total_columns': analysis['total_columns'],
            'numeric_columns': analysis['numeric_columns'],
            'categorical_columns': analysis['categorical_columns'],
            'date_columns': analysis['date_columns']
        }
        
        # Add specific insights based on data
        insights = []
        
        # Numeric column insights
        for col in analysis['numeric_columns'][:3]:  # Limit to first 3 columns
            if col in analysis['data_ranges']:
                range_info = analysis['data_ranges'][col]
                insights.append(f"{col}: min={range_info['min']:.2f}, max={range_info['max']:.2f}, avg={range_info['mean']:.2f}")
        
        # Categorical column insights
        for col in analysis['categorical_columns'][:3]:  # Limit to first 3 columns
            if col in analysis['unique_counts']:
                insights.append(f"{col}: {analysis['unique_counts'][col]} unique values")
        
        # Language-specific prompts
        if lang == 'en':
            prompt = f"""
            Generate a comprehensive explanation for a data visualization based on the following information:

            User Question: "{question}"
            Visualization Type: {viz_type}
            
            Data Summary:
            - Total Records: {data_summary['total_rows']:,}
            - Total Columns: {data_summary['total_columns']}
            - Numeric Columns: {', '.join(data_summary['numeric_columns']) if data_summary['numeric_columns'] else 'None'}
            - Categorical Columns: {', '.join(data_summary['categorical_columns']) if data_summary['categorical_columns'] else 'None'}
            - Date/Time Columns: {', '.join(data_summary['date_columns']) if data_summary['date_columns'] else 'None'}
            
            Key Data Insights:
            {chr(10).join(f"- {insight}" for insight in insights) if insights else '- No specific insights available'}
            
            Please provide a detailed explanation that includes:
            1. Why this visualization type was chosen
            2. What the chart shows and how to interpret it
            3. Key insights from the data
            4. How this answers the user's question
            
            Make the explanation clear, informative, and easy to understand.
            """
        
        elif lang == 'id':
            prompt = f"""
            Buatkan penjelasan komprehensif untuk visualisasi data berdasarkan informasi berikut:

            Pertanyaan Pengguna: "{question}"
            Jenis Visualisasi: {viz_type}
            
            Ringkasan Data:
            - Total Record: {data_summary['total_rows']:,}
            - Total Kolom: {data_summary['total_columns']}
            - Kolom Numerik: {', '.join(data_summary['numeric_columns']) if data_summary['numeric_columns'] else 'Tidak ada'}
            - Kolom Kategori: {', '.join(data_summary['categorical_columns']) if data_summary['categorical_columns'] else 'Tidak ada'}
            - Kolom Tanggal/Waktu: {', '.join(data_summary['date_columns']) if data_summary['date_columns'] else 'Tidak ada'}
            
            Wawasan Data Utama:
            {chr(10).join(f"- {insight}" for insight in insights) if insights else '- Tidak ada wawasan spesifik'}
            
            Berikan penjelasan detail yang mencakup:
            1. Mengapa jenis visualisasi ini dipilih
            2. Apa yang ditampilkan grafik dan cara membacanya
            3. Wawasan utama dari data
            4. Bagaimana ini menjawab pertanyaan pengguna
            
            Buat penjelasan yang jelas, informatif, dan mudah dipahami dalam Bahasa Indonesia.
            """
        
        else:
            # Generic prompt for other languages
            lang_names = {
                'es': 'Spanish',
                'fr': 'French', 
                'de': 'German',
                'zh': 'Chinese',
                'ja': 'Japanese',
                'ko': 'Korean',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ar': 'Arabic'
            }
            
            lang_name = lang_names.get(lang, lang.upper())
            
            prompt = f"""
            Generate a comprehensive data visualization explanation in {lang_name} based on:

            User Question: "{question}"
            Visualization Type: {viz_type}
            
            Data Summary:
            - Total Records: {data_summary['total_rows']:,}
            - Columns: {data_summary['total_columns']}
            - Numeric: {', '.join(data_summary['numeric_columns']) if data_summary['numeric_columns'] else 'None'}
            - Categorical: {', '.join(data_summary['categorical_columns']) if data_summary['categorical_columns'] else 'None'}
            - Date/Time: {', '.join(data_summary['date_columns']) if data_summary['date_columns'] else 'None'}
            
            Data Insights: {', '.join(insights) if insights else 'No specific insights'}
            
            Provide explanation in {lang_name} covering:
            1. Why this visualization was chosen
            2. How to interpret the chart
            3. Key data insights
            4. How it answers the user's question
            
            Keep technical terms (like SELECT, WHERE) in English but explain everything else in {lang_name}.
            """
        
        # Generate explanation using Ollama
        explanation = ollama_llm.invoke(
            prompt,
            temperature=0.3,  # Slightly higher for more natural language
            top_p=0.9,
            repeat_penalty=1.2
        )
        
        return explanation.strip()
        
    except Exception as e:
        print(f"LLM Visualization explanation error: {e}")
        
        # Fallback explanation
        fallback_messages = {
            'en': f"""
            **Visualization Analysis**
            
            Chart Type: {viz_type}
            Data Records: {len(df):,}
            Columns: {len(df.columns)}
            
            This visualization was generated to help answer your question: "{question}"
            
            The chart displays the data in a format that makes patterns and insights easier to understand.
            """,
            'id': f"""
            **Analisis Visualisasi**
            
            Jenis Grafik: {viz_type}
            Record Data: {len(df):,}
            Kolom: {len(df.columns)}
            
            Visualisasi ini dibuat untuk membantu menjawab pertanyaan Anda: "{question}"
            
            Grafik menampilkan data dalam format yang membuat pola dan wawasan lebih mudah dipahami.
            """
        }
        
        return fallback_messages.get(lang, fallback_messages['en'])

def generate_visualization_with_explanation(df: pd.DataFrame, question: str, lang: str = 'en') -> Tuple[Optional[str], str]:
    """
    Generate visualization and comprehensive explanation using LLM
    """
    try:
        # Analyze data structure
        analysis = analyze_data_for_visualization(df)
        
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        date_cols = analysis['date_columns']
        
        fig = None
        viz_type = ""
        
        # Determine best visualization type and create chart
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            # Time series line chart
            viz_type = "Time Series Line Chart"
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], 
                         title=f"{numeric_cols[0]} over {date_cols[0]}")
            fig.update_layout(
                xaxis_title=date_cols[0].replace('_', ' ').title(),
                yaxis_title=numeric_cols[0].replace('_', ' ').title(),
                template="plotly_white"
            )
            
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Bar chart for categorical vs numeric
            viz_type = "Bar Chart"
            cat_col, num_col = categorical_cols[0], numeric_cols[0]
            
            # Group and aggregate data (limit to top 20 categories to avoid clutter)
            grouped_df = df.groupby(cat_col)[num_col].sum().reset_index()
            grouped_df = grouped_df.nlargest(20, num_col)
            
            fig = px.bar(grouped_df, x=cat_col, y=num_col, 
                        title=f"{num_col} by {cat_col}")
            fig.update_layout(
                xaxis_title=cat_col.replace('_', ' ').title(),
                yaxis_title=num_col.replace('_', ' ').title(),
                template="plotly_white"
            )
            
        elif len(numeric_cols) >= 2:
            # Scatter plot for numeric correlations
            viz_type = "Scatter Plot"
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            fig = px.scatter(df, x=x_col, y=y_col, 
                           title=f"Correlation: {x_col} vs {y_col}")
            fig.update_layout(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                template="plotly_white"
            )
            
        elif len(categorical_cols) >= 1:
            # Pie chart for categorical distribution (limit to top 10 categories)
            viz_type = "Pie Chart"
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().head(10)
            
            fig = px.pie(names=value_counts.index, values=value_counts.values, 
                        title=f"Distribution of {cat_col}")
            fig.update_layout(template="plotly_white")
            
        else:
            # Simple count visualization
            viz_type = "Data Count Overview"
            fig = go.Figure(go.Bar(
                x=["Total Records"],
                y=[len(df)],
                text=[f"{len(df):,}"],
                textposition='auto',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title='Data Count Overview',
                xaxis_title="Metric",
                yaxis_title="Count",
                template="plotly_white"
            )
        
        # Generate explanation using LLM
        explanation = generate_llm_visualization_explanation(df, question, viz_type, lang)
        
        # Convert chart to base64
        if fig:
            buf = io.BytesIO()
            fig.write_image(buf, format='png', width=800, height=600, scale=2)
            visualization_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return visualization_b64, explanation
        else:
            return None, explanation
            
    except Exception as e:
        print(f"Visualization generation error: {e}")
        
        # Fallback visualization and explanation
        try:
            fallback_fig = go.Figure(go.Bar(
                x=["Data Records"],
                y=[len(df)],
                text=[f"{len(df):,}"],
                textposition='auto',
                marker_color='lightcoral'
            ))
            fallback_fig.update_layout(
                title='Data Overview (Fallback)',
                template="plotly_white"
            )
            
            buf = io.BytesIO()
            fallback_fig.write_image(buf, format='png', width=800, height=600)
            fallback_viz = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Generate fallback explanation using LLM
            fallback_explanation = generate_llm_visualization_explanation(
                df, question, "Data Count Overview", lang
            )
            
            return fallback_viz, fallback_explanation
            
        except Exception as fallback_error:
            print(f"Fallback visualization failed: {fallback_error}")
            error_messages = {
                'en': f"Unable to generate visualization. Error: {str(e)}",
                'id': f"Tidak dapat membuat visualisasi. Error: {str(e)}"
            }
            return None, error_messages.get(lang, error_messages['en'])

def get_sql_explanation(sql: str, lang: str) -> str:
    """Generate SQL explanation in the specified language with better accuracy"""
    
    if lang == 'en':
        prompt = f"""
        Explain this SQL query clearly and concisely:
        
        {sql}
        
        Focus on:
        1. Purpose of the query
        2. Tables and columns involved
        3. Conditions and filters
        4. Expected results
        
        Be accurate and don't add information not present in the query.
        """
    else:
        # Language-specific prompts for better accuracy
        lang_names = {
            'id': 'Indonesian',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese'
        }
        
        lang_name = lang_names.get(lang, lang.upper())
        
        prompt = f"""
        Explain this SQL query in {lang_name}. Be accurate and don't invent details.
        
        SQL Query: {sql}
        
        Provide explanation in {lang_name} with this structure:
        1. Query purpose (what it does)
        2. Tables/columns used
        3. Conditions or filters applied
        4. Expected output format
        
        Keep technical SQL terms in English (SELECT, WHERE, JOIN, etc.) but explain in {lang_name}.
        Do not add assumptions or data not shown in the actual query.
        """
    
    try:
        explanation = ollama_llm.invoke(
            prompt,
            temperature=0.1,  # Lower temperature for more accurate responses
            top_p=0.9,
            repeat_penalty=1.2
        )
        return explanation.strip()
    except Exception as e:
        print(f"Explanation error: {e}")
        return f"**SQL Explanation Error**: Could not generate explanation in {lang}"

def analyze_data_for_visualization(df: pd.DataFrame) -> Dict:
    """
    Analyze DataFrame to determine the best visualization type and gather insights
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'date_columns': date_cols,
        'null_counts': df.isnull().sum().to_dict(),
        'data_ranges': {},
        'unique_counts': {}
    }
    
    # Get data ranges for numeric columns
    for col in numeric_cols:
        analysis['data_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median())
        }
    
    # Get unique counts for categorical columns
    for col in categorical_cols:
        analysis['unique_counts'][col] = len(df[col].unique())
    
    return analysis

def convert_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all data types in DataFrame to JSON-serializable formats
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Get the column data
        col_data = df_copy[col]
        
        # Handle different data types
        if col_data.dtype == 'object':
            # Check what type of objects are in this column
            sample_value = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
            
            if sample_value is not None:
                # Date objects
                if isinstance(sample_value, date) and not isinstance(sample_value, datetime):
                    df_copy[col] = col_data.apply(
                        lambda x: x.isoformat() if isinstance(x, date) and pd.notnull(x) else None
                    )
                # Time objects
                elif isinstance(sample_value, time):
                    df_copy[col] = col_data.apply(
                        lambda x: x.isoformat() if isinstance(x, time) and pd.notnull(x) else None
                    )
                # Decimal objects
                elif isinstance(sample_value, Decimal):
                    df_copy[col] = col_data.apply(
                        lambda x: float(x) if isinstance(x, Decimal) and pd.notnull(x) else None
                    )
                # UUID objects
                elif hasattr(sample_value, 'hex'):  # UUID check
                    df_copy[col] = col_data.apply(
                        lambda x: str(x) if pd.notnull(x) else None
                    )
                # Other objects that might need string conversion
                elif not isinstance(sample_value, (str, int, float, bool)):
                    df_copy[col] = col_data.apply(
                        lambda x: str(x) if pd.notnull(x) else None
                    )
        
        # Handle datetime64 columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            df_copy[col] = col_data.apply(
                lambda x: x.isoformat() if pd.notnull(x) else None
            )
        
        # Handle timedelta columns
        elif pd.api.types.is_timedelta64_dtype(col_data):
            df_copy[col] = col_data.dt.total_seconds()
        
        # Handle numeric columns with potential issues
        elif pd.api.types.is_numeric_dtype(col_data):
            # Convert to float, handling NaN properly
            df_copy[col] = col_data.astype('float64')
            # Replace inf and -inf with None
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], None)
        
        # Handle boolean columns
        elif pd.api.types.is_bool_dtype(col_data):
            # Convert to standard Python bool
            df_copy[col] = col_data.astype('bool')
        
        # Handle categorical columns
        elif pd.api.types.is_categorical_dtype(col_data):
            df_copy[col] = col_data.astype(str)
    
    return df_copy

def safe_json_serialize(data):
    """
    Custom JSON serializer for problematic data types
    """
    if isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, time):
        return data.isoformat()
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif pd.isna(data) or data is None:
        return None
    elif isinstance(data, (np.inf, -np.inf)):
        return None
    else:
        return str(data)

def process_sql_results(result_df: pd.DataFrame) -> list:
    """
    Process SQL results and convert to JSON-serializable format
    """
    if result_df.empty:
        return []
    
    try:
        # Convert problematic data types
        processed_df = convert_dataframe_for_json(result_df)
        
        # Convert to dictionary records
        result_data = processed_df.to_dict('records')
        
        # Final safety check - serialize and deserialize to catch any remaining issues
        json_str = json.dumps(result_data, default=safe_json_serialize)
        return json.loads(json_str)
        
    except Exception as e:
        print(f"Error processing SQL results: {e}")
        # Fallback: convert everything to strings
        fallback_df = result_df.copy()
        for col in fallback_df.columns:
            fallback_df[col] = fallback_df[col].apply(
                lambda x: str(x) if pd.notnull(x) else None
            )
        return fallback_df.to_dict('records')

# def classify_question_with_fallback(question_text):
#     """
#     Enhanced classification with keyword-based fallback
#     """
#     question_lower = question_text.lower()
    
#     # Direct keyword matching for common cases
#     dca_keywords = ['dca', 'decline curve', 'declive curve analysis', 'production forecast', 'decline analysis', 'production decline']
#     wellbore_keywords = ['wellbore', 'well diagram', 'completion', 'casing', 'tubing', 'perforation', 'well component', 'components', 'wellbore components']
    
#     # Check for DCA keywords
#     if any(keyword in question_lower for keyword in dca_keywords):
#         return 'dca'
    
#     # Check for wellbore keywords  
#     if any(keyword in question_lower for keyword in wellbore_keywords):
#         return 'wellbore'
    
#     # If no direct keywords found, use LLM classification
#     try:
#         classification = ollama_llm.invoke(
#             classification_prompt,
#             temperature=0.0,
#             top_p=1.0,
#             repeat_penalty=1.0
#         ).strip().lower()
        
#         # Validate the classification response
#         valid_classifications = ['dca', 'wellbore', 'none']
#         if classification in valid_classifications:
#             return classification
#         else:
#             return 'none'  # Default fallback
            
#     except Exception as e:
#         print(f"Classification error: {e}")
#         return 'none'  # Default fallback

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            question_data = json.loads(data)
            
            # Detect language
            original_lang = detect_language(question_data['question'])
            
            # Translate if needed
            english_question = question_data['question']
            if original_lang != 'en':
                english_question = translate_to_english(question_data['question'])
            
            try:
                # Generate SQL
                sql = vn_milvus.generate_sql(english_question)
                
                if not sql:
                    error_messages = {
                        'en': "Could not generate SQL for your question",
                        'id': "Tidak dapat menghasilkan SQL untuk pertanyaan Anda"
                    }
                    await websocket.send_json({
                        "type": "error",
                        "message": error_messages.get(original_lang, error_messages['en'])
                    })
                    continue
                
                # Get SQL explanation
                explanation = get_sql_explanation(sql, original_lang)
                
                # Execute SQL and process results
                result_df = vn_milvus.run_sql(sql)
                result_data = process_sql_results(result_df)
                
                # Generate enhanced visualization with LLM explanation
                visualization, viz_explanation = generate_visualization_with_explanation(
                    result_df, question_data['question'], original_lang
                )
                
                # Send response
                response = {
                    "type": "success",
                    "sql": sql,
                    "explanation": explanation,
                    "data": result_data,
                    "original_question": question_data['question'],
                    "detected_language": original_lang
                }
                
                if english_question != question_data['question']:
                    response["translated_question"] = english_question
                
                if visualization:
                    response["visualization"] = visualization
                    response["visualization_explanation"] = viz_explanation
                else:
                    response["visualization_explanation"] = viz_explanation
                
                await websocket.send_json(response)
                
            except Exception as e:
                print(f"WebSocket Error: {e}")
                error_messages = {
                    'en': f"An error occurred: {str(e)}",
                    'id': f"Terjadi kesalahan: {str(e)}"
                }
                await websocket.send_json({
                    "type": "error",
                    "message": error_messages.get(original_lang, error_messages['en'])
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/api/query")
async def query(question: Question):
    try:
        # Previous code for SQL generation and execution...
        original_lang = detect_language(question.question)
        original_question = question.question
        
        english_question = original_question
        if original_lang != 'en':
            english_question = translate_to_english(original_question)
        
        # Simple keyword-based classification (no LLM needed)
        def classify_question(question_text):
            question_lower = question_text.lower()
            
            # DCA-related keywords
            dca_keywords = ['dca', 'decline curve', 'production forecast', 'decline analysis', 'production decline']
            if any(keyword in question_lower for keyword in dca_keywords):
                return 'dca'
            
            # Wellbore-related keywords  
            wellbore_keywords = ['wellbore', 'well diagram', 'completion', 'casing', 'tubing', 'perforation', 'well component']
            if any(keyword in question_lower for keyword in wellbore_keywords):
                return 'wellbore'
            
            return 'none'
        
        classification = classify_question(english_question)
        
        # Add debug logging
        print(f"Question: {english_question}")
        print(f"Classification: {classification}")
        
        sql = vn_milvus.generate_sql(english_question)
        
        if not sql:
            error_messages = {
                'en': "Could not generate SQL for your question",
                'id': "Tidak dapat menghasilkan SQL untuk pertanyaan Anda"
            }
            error_msg = error_messages.get(original_lang, error_messages['en'])
            
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
        
        explanation = get_sql_explanation(sql, original_lang)
        result_df = vn_milvus.run_sql(sql)
        result_data = process_sql_results(result_df)
        
        # Generate enhanced visualization with explanation
        visualization, viz_explanation = generate_visualization_with_explanation(
            result_df, original_question, original_lang
        )
        
        response = {
            "type": "success",
            "sql": sql,
            "explanation": explanation,
            "data": result_data,
            "original_question": original_question,
            "detected_language": original_lang
        }
        
        # Conditionally inject the app URL based on classification
        if classification == "dca":
            response["app_url"] = "https://syauqialzaa.github.io/dca/"
            print("Added DCA URL")
        elif classification == "wellbore":
            response["app_url"] = "https://syauqialzaa.github.io/wellbore/"
            print("Added Wellbore URL")
        
        if english_question != original_question:
            response["translated_question"] = english_question
        
        if visualization:
            response["visualization"] = visualization
            response["visualization_explanation"] = viz_explanation
        else:
            response["visualization_explanation"] = viz_explanation
            
        return response
        
    except Exception as e:
        print(f"API Query Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
