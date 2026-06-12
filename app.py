import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from nlp_engine import SentimentEngine

# Initialize the ML/NLP sentiment engine
engine = SentimentEngine()

app = FastAPI(
    title="AI-Based Sentiment Analysis Dashboard API",
    description="Backend API for customer feedback preprocessing, scoring, and bulk exports.",
    version="1.0.0"
)

# Ensure temporary exports directory exists within the workspace
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Define request schema for single text analysis
class SingleTextRequest(BaseModel):
    text: str

# Serve the static UI files from the /static directory
# Check if the static directory exists, if not it will be created in the next steps
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

@app.post("/api/analyze/single")
async def analyze_single_text(payload: SingleTextRequest):
    """
    Endpoint for real-time text analysis. Returns cleaned text stages,
    VADER scores, and Machine Learning classifications.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
    try:
        analysis = engine.analyze_single(payload.text)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze/bulk")
async def analyze_bulk_csv(
    file: UploadFile = File(...),
    column: str = Form(None)
):
    """
    Endpoint for bulk feedback analysis. Accepts a CSV file, identifies the text column,
    runs classifications, computes statistics, and generates a downloadable CSV.
    """
    # Verify file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        # Save temporary uploaded file contents to read into pandas
        contents = await file.read()
        
        # Parse CSV
        try:
            # Try parsing with UTF-8 first, fallback to ISO-8859-1 if needed
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        except UnicodeDecodeError:
            from io import BytesIO
            df = pd.read_csv(BytesIO(contents), encoding='ISO-8859-1')
            
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
            
        # Determine target text column
        selected_column = column
        if not selected_column:
            # Auto-detect column based on common naming conventions
            candidates = []
            for col in df.columns:
                col_lower = col.lower()
                # Exclude columns ending with ID or containing ID/date/time
                if col_lower.endswith('id') or '_id' in col_lower or 'date' in col_lower or 'time' in col_lower:
                    continue
                # Score columns based on keywords
                score = 0
                if 'text' in col_lower: score += 10
                if 'review' in col_lower: score += 10
                if 'comment' in col_lower: score += 10
                if 'feedback' in col_lower: score += 8
                if 'message' in col_lower: score += 8
                if 'body' in col_lower: score += 8
                if 'content' in col_lower: score += 8
                
                if score > 0:
                    candidates.append((col, score))
            
            if candidates:
                # Pick the highest scoring candidate
                selected_column = max(candidates, key=lambda x: x[1])[0]
            else:
                # Fallback: find first column that is not ID-like
                for col in df.columns:
                    col_lower = col.lower()
                    if not (col_lower.endswith('id') or '_id' in col_lower or 'date' in col_lower or 'time' in col_lower):
                        selected_column = col
                        break
                # Ultimate fallback to the first column
                if not selected_column:
                    selected_column = df.columns[0]
                
        # Validate column existence
        if selected_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{selected_column}' not found. Available columns: {list(df.columns)}"
            )
            
        # Perform sentiment batch analysis
        analyzed_df, summary = engine.analyze_batch(df, selected_column)
        
        # Merge analyzed columns back to the original dataframe to preserve all upload data
        for col in ["Cleaned_Text", "VADER_Compound", "VADER_Sentiment", "ML_Sentiment"]:
            df[col] = analyzed_df[col]
            
        # Write processed file to the exports directory
        job_id = str(uuid.uuid4())
        export_filename = f"{job_id}.csv"
        export_filepath = os.path.join(EXPORT_DIR, export_filename)
        df.to_csv(export_filepath, index=False)
        
        # Add job info to summary
        summary["job_id"] = job_id
        summary["detected_column"] = selected_column
        summary["available_columns"] = list(df.columns[:-4]) # Exclude the new columns
        
        return summary
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")

@app.get("/api/export/{job_id}")
async def export_job_results(job_id: str):
    """
    Serves the labeled CSV output for download.
    """
    file_path = os.path.join(EXPORT_DIR, f"{job_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found or expired.")
        
    return FileResponse(
        path=file_path, 
        media_type="text/csv", 
        filename="analyzed_customer_feedback.csv"
    )

@app.get("/api/model-info")
async def get_model_information():
    """
    Returns statistics and key features of the active ML sentiment classifier.
    """
    try:
        model_details = engine.get_model_features()
        return model_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(e)}")

# Fallback route to serve frontend index.html on root access
@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "FastAPI is running! Static index.html not found."})

# Mount the static directory for script.js, style.css, etc.
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
