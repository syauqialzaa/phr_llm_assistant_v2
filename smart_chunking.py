import time
from collections import defaultdict
from datetime import datetime, timedelta
import json
import random
import pandas as pd
from pymilvus import MilvusClient, model
from vanna.milvus import Milvus_VectorStore
from vanna.ollama import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

class VannaMilvus(Milvus_VectorStore, Ollama):
    def __init__(self, config=None):
        Milvus_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
    
    def safe_ask(self, question):
        try:
            sql = self.generate_sql(question)
            
            if not sql or not sql.strip():
                return "I'm sorry, I couldn't understand your question. Please try rephrasing it in the context of the database."
            
            print(f"\nGenerated SQL: {sql}")
            
            try:
                self.validate_sql(sql)
                
                result = self.run_sql(sql)
                return result
            except Exception as e:
                return f"I generated SQL but it seems to have an error: {str(e)}. Please try rephrasing your question."
            
        except Exception as e:
            return f"I'm unable to process your question. It might be outside my current knowledge. Error: {str(e)}"

    def validate_sql(self, sql):
        validate_sql = f"EXPLAIN {sql}"
        try:
            self.run_sql(validate_sql)
        except Exception as e:
            raise ValueError(f"SQL validation failed: {str(e)}")

vn_milvus = VannaMilvus(
    config={
        "model": "mistral-nemo",
        "embedding_function": model.DefaultEmbeddingFunction(),
        "n_results": 2,  # The number of results to return from Milvus semantic search.
    }
)

vn_milvus.connect_to_postgres(
    host=os.getenv("PG_HOST"),
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=os.getenv("PG_PORT")
)

df_information_schema = vn_milvus.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
plan = vn_milvus.get_training_plan_generic(df_information_schema)
vn_milvus.train(plan=plan)

def validate_training_row(row, sheet_type):
    if sheet_type == "DDL":
        return pd.notna(row.get("DDL")) and any(kw in row["DDL"].upper() for kw in ["CREATE", "ALTER", "DROP"])
    elif sheet_type == "Documentation":
        return pd.notna(row.get("Documentation")) and len(row["Documentation"]) > 20
    elif sheet_type == "SQL":
        return pd.notna(row.get("SQL")) and any(kw in row["SQL"].upper() for kw in ["SELECT", "INSERT", "UPDATE"])
    elif sheet_type == "Question-SQL":
        return pd.notna(row.get("Question")) and pd.notna(row.get("SQL"))
    return False

# helper function to add metadata as a comment to the training text.
def add_metadata_to_text(text, metadata):
    # convert metadata to a json string and add as a comment (works for SQL or DDL)
    metadata_comment = "\n-- metadata: " + json.dumps(metadata)
    return text + metadata_comment

def generate_chunk_metadata(row, sheet_type):
    metadata = {
        "sheet_type": sheet_type,
        "row_hash": hash(str(row.to_dict())),
        "timestamp": datetime.now().isoformat(),
        "chunk_strategy": "smart_chunking"
    }
    if sheet_type == "Question-SQL":
        metadata.update({
            "question_length": len(row.get("Question", "")),
            "sql_complexity": len(row.get("SQL", "").split())
        })
    return metadata

def train_with_smart_chunking(vn_instance, excel_path):
    """smart chunking training implementation"""
    chunk_counter = defaultdict(int)
    error_log = []
    
    # for each sheet, lambda created that adds the metadata to the content before training.
    sheet_config = {
        "DDL": {
            "column": "DDL",
            "train_method": lambda row: vn_instance.train(
                ddl=add_metadata_to_text(row["DDL"], generate_chunk_metadata(row, "DDL"))
            )
        },
        "Documentation": {
            "column": "Documentation",
            "train_method": lambda row: vn_instance.train(
                documentation=add_metadata_to_text(row["Documentation"], generate_chunk_metadata(row, "Documentation"))
            )
        },
        "SQL": {
            "column": "SQL",
            "train_method": lambda row: vn_instance.train(
                sql=add_metadata_to_text(row["SQL"], generate_chunk_metadata(row, "SQL"))
            )
        },
        "Question-SQL": {
            "columns": ["Question", "SQL"],
            "train_method": lambda row: vn_instance.train(
                question=add_metadata_to_text(row["Question"], generate_chunk_metadata(row, "Question-SQL")),
                sql=add_metadata_to_text(row["SQL"], generate_chunk_metadata(row, "Question-SQL"))
            )
        }
    }

    for sheet_name, config in sheet_config.items():
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            for idx, row in df.iterrows():
                try:
                    if validate_training_row(row, sheet_name):
                        config["train_method"](row)
                        chunk_counter[sheet_name] += 1
                        time.sleep(0.05)  # Rate limiting
                    else:
                        error_log.append(f"Invalid row {idx+1} in {sheet_name}")
                except Exception as e:
                    error_log.append(f"Error in {sheet_name} row {idx+1}: {str(e)}")
        except Exception as e:
            error_log.append(f"Sheet {sheet_name} error: {str(e)}")

    print("\n=== Training Report ===")
    print(f"Total chunks trained: {sum(chunk_counter.values())}")
    for sheet, count in chunk_counter.items():
        print(f"- {sheet}: {count} chunks")
    
    if error_log:
        print("\n=== Errors ===")
        for error in error_log[:10]:
            print(f"- {error}")

excel_file = "phr_api_training_data.xlsx"
train_with_smart_chunking(vn_milvus, excel_file)

training_data = vn_milvus.get_training_data()
training_data

if __name__ == "__main__":
    print("\n=== Query Interface ===")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
            
            response = vn_milvus.safe_ask(question)
            
            if isinstance(response, pd.DataFrame):
                print(f"\nResults ({len(response)} rows):")
                print(response.head().to_markdown(index=False))
            else:
                print(f"\nResponse: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

    print("\nSession ended. Goodbye!")

