"""
FastAPI endpoint that loads the trained PySpark ML PipelineModels
and predicts Student Headcounts based on the given parameters.
"""

import json
import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\mathi\anaconda3\envs\waterloo\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\mathi\anaconda3\envs\waterloo\python.exe"
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expm1, greatest, lit, monotonically_increasing_id, substring
from pyspark.ml import PipelineModel


# ── Schema for the prediction request ────────────────────────────────────────

class PredictionRequest(BaseModel):
    fiscal_year: str              # e.g. "2023/24"
    term_type: str                # e.g. "Fall term"
    career: str                   # e.g. "Undergraduate"
    program_level: str            # e.g. "Bachelors"
    study_year: str               # e.g. "1"
    campus: str                   # e.g. "University of Waterloo"
    faculty_group: str            # e.g. "ENG"
    program_grouping: str         # e.g. "Computer Science"
    coop_regular: str             # e.g. "Co-op"
    work_term: str                # e.g. "Academic Term"
    attendance: str               # e.g. "Full-Time"
    visa_status: str              # e.g. "Canadian"

    model_config = {"json_schema_extra": {
        "examples": [{
            "fiscal_year": "2023/24",
            "term_type": "Fall term",
            "career": "Undergraduate",
            "program_level": "Bachelors",
            "study_year": "3",
            "campus": "University of Waterloo",
            "faculty_group": "MATH",
            "program_grouping": "Computer Science",
            "coop_regular": "Co-op",
            "work_term": "Academic Term",
            "attendance": "Full-Time",
            "visa_status": "Canadian",
        }]
    }}


class PredictionResponse(BaseModel):
    model_name: str
    predicted_student_headcount: float


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]


# ── Application state (populated at startup) ─────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Spark session and models once at startup, clean up on shutdown."""
    os.environ.setdefault(
        "JAVA_HOME",
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
    )

    spark = (
        SparkSession.builder
        .appName("UW Enrolement API")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.extraJavaOptions", "-Xss4m")
        .config("spark.executor.extraJavaOptions", "-Xss4m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load model metadata
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(BASE_DIR, "models", "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    best_model_name = metadata["best_model"]
    models_info = metadata["models"]

    # Load all saved PipelineModels
    loaded_models = {}
    for name, info in models_info.items():
        clean_path = info["path"].replace("/", "\\")
        model_path = os.path.join(BASE_DIR, clean_path)
        
        print("Loading model from:", model_path)  # debug
        loaded_models[name] = PipelineModel.load(model_path)

    _state["spark"] = spark
    _state["metadata"] = metadata
    _state["models"] = loaded_models
    _state["best_model_name"] = best_model_name

    yield  # app is running

    spark.stop()


# ── Create the app ───────────────────────────────────────────────────────────

app = FastAPI(
    title="UW Enrolment Prediction API",
    description="Predict student headcounts using trained PySpark ML models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    """Return available models and their evaluation metrics."""
    return _state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, model_name: Optional[str] = None):
    
    year = int(req.fiscal_year.split("/")[0])
    
    """
    Predict Student Headcounts.

    If *model_name* is omitted the best model (by R²) is used.
    Pass one of the model names returned by ``GET /models`` to pick
    a specific model.
    """
    chosen = model_name or _state["best_model_name"]

    if chosen not in _state["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{chosen}'. "
                   f"Available: {list(_state['models'].keys())}",
        )

    spark: SparkSession = _state["spark"]
    pipeline_model: PipelineModel = _state["models"][chosen]

    # Build a single-row Spark DataFrame that matches the training schema
    row = {
        "Fiscal Year": year,
        "Term Type": req.term_type,
        "Career": req.career,
        "Program Level": req.program_level,
        "Study Year": req.study_year,
        "Campus": req.campus,
        "Faculty (group)": req.faculty_group,
        "Program Grouping": req.program_grouping,
        "Coop Regular": req.coop_regular,
        "Work Term": req.work_term,
        "Attendance": req.attendance,
        "Visa Status": req.visa_status,
    }

    input_df = spark.createDataFrame([row])

    # Add the numeric_year feature (first 4 chars of Fiscal Year cast to double)
    input_df = input_df.withColumn(
        "numeric_year",
        substring(col("Fiscal Year"), 1, 4).cast("double"),
    )

    # The PipelineModel already includes all preprocessing stages
    # (StringIndexer + OneHotEncoder/VectorAssembler), so we can
    # call transform() directly on raw input.
    predictions = pipeline_model.transform(input_df)

    # Inverse the log1p transform and clip to a minimum of 1
    predictions = predictions.withColumn(
        "final_prediction",
        greatest(expm1(col("prediction")), lit(1.0)),
    )

    result = predictions.select("final_prediction").head()
    predicted_count = float(result["final_prediction"])

    return PredictionResponse(
        model_name=chosen,
        predicted_student_headcount=round(predicted_count, 2),
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictionRequest, model_name: Optional[str] = None):
    """
    Batch-predict Student Headcounts for multiple parameter combinations.

    Accepts a list of prediction requests and returns all predictions
    in a single response.  Much more efficient than calling ``/predict``
    in a loop because only one Spark transform is executed.
    """
    if not req.requests:
        return {"model_name": model_name or _state["best_model_name"], "predictions": []}

    chosen = model_name or _state["best_model_name"]

    if chosen not in _state["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{chosen}'. "
                   f"Available: {list(_state['models'].keys())}",
        )

    spark: SparkSession = _state["spark"]
    pipeline_model: PipelineModel = _state["models"][chosen]

    rows = [
        {
            "Fiscal Year": r.fiscal_year,
            "Term Type": r.term_type,
            "Career": r.career,
            "Program Level": r.program_level,
            "Study Year": r.study_year,
            "Campus": r.campus,
            "Faculty (group)": r.faculty_group,
            "Program Grouping": r.program_grouping,
            "Coop Regular": r.coop_regular,
            "Work Term": r.work_term,
            "Attendance": r.attendance,
            "Visa Status": r.visa_status,
        }
        for r in req.requests
    ]

    input_df = spark.createDataFrame(rows)
    input_df = input_df.withColumn(
        "numeric_year",
        substring(col("Fiscal Year"), 1, 4).cast("double"),
    )
    input_df = input_df.withColumn("_idx", monotonically_increasing_id())

    predictions = pipeline_model.transform(input_df)
    predictions = predictions.withColumn(
        "final_prediction",
        greatest(expm1(col("prediction")), lit(1.0)),
    )

    result_rows = (
        predictions
        .orderBy("_idx")
        .select(
            "Fiscal Year", "Term Type", "Career", "Program Level",
            "Study Year", "Campus", "Faculty (group)", "Program Grouping",
            "Coop Regular", "Work Term", "Attendance", "Visa Status",
            "final_prediction",
        )
        .collect()
    )

    return {
        "model_name": chosen,
        "predictions": [
            {
                "fiscal_year": row["Fiscal Year"],
                "term_type": row["Term Type"],
                "career": row["Career"],
                "program_level": row["Program Level"],
                "study_year": row["Study Year"],
                "campus": row["Campus"],
                "faculty_group": row["Faculty (group)"],
                "program_grouping": row["Program Grouping"],
                "coop_regular": row["Coop Regular"],
                "work_term": row["Work Term"],
                "attendance": row["Attendance"],
                "visa_status": row["Visa Status"],
                "predicted_student_headcount": round(float(row["final_prediction"]), 2),
            }
            for row in result_rows
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
