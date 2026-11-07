import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("Mental_Health_Burnout.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class PersonData(BaseModel):
    gender: Literal['Male', 'Female', 'Non-binary', 'Prefer not to say']
    country: Literal['Australia', 'Brazil', 'Canada', 'France', 'Germany',
                    'India', 'Netherlands', 'Singapore', 'UK', 'USA']
    job_role: Literal['Software Engineer', 'Data Scientist', 'ML Engineer', 'Data Analyst',
                      'Backend Developer', 'Frontend Developer', 'Product Manager',
                      'DevOps Engineer', 'Full Stack Developer', 'Cybersecurity Engineer',
                      'Cloud Engineer', 'QA Engineer']
    seniority_level: Literal['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Principal']
    years_experience: int = Field(..., ge=0, le=30)
    years_at_company: float = Field(..., ge=0.0, le=16.0)
    company_size: Literal['Startup (1-50)', 'Small (51-200)', 'Mid (201-1000)',
                          'Large (1001-5000)', 'Enterprise (5000+)']
    industry: Literal['SaaS / Cloud', 'Fintech', 'E-commerce', 'AI / ML Startup',
                      'Healthcare Tech', 'Cybersecurity', 'Consulting',
                      'Enterprise Software', 'Gaming', 'Social Media / AdTech']
    work_mode: Literal['Hybrid', 'Remote', 'On-site']
    salary_usd: int = Field(..., ge=40000, le=270000)
    work_hours_per_week: int = Field(..., ge=35, le=72)
    meetings_per_day: float = Field(..., ge=0.0, le=12.0)
    team_size: int = Field(..., ge=2, le=59)
    sleep_hours_per_night: float = Field(..., ge=3.0, le=10.0)
    exercise_days_per_week: int = Field(..., ge=0, le=7)
    vacation_days_taken: int = Field(..., ge=0, le=30)
    therapy_access: Literal[0, 1]
    uses_therapy: Literal[0, 1]
    ai_tools_daily: Literal[0, 1]
    manager_support_score: float = Field(..., ge=1.0, le=10.0)
    work_life_balance_score: float = Field(..., ge=1.0, le=10.0)
    job_satisfaction_score: float = Field(..., ge=1.0, le=10.0)
    social_support_score: float = Field(..., ge=1.0, le=10.0)
    deadline_pressure_score: float = Field(..., ge=1.0, le=10.0)
    autonomy_score: float = Field(..., ge=1.0, le=10.0)
    seeks_mental_health_support: Literal[0, 1]
    job_change_intention: Literal[0, 1]


class PredictionResponse(BaseModel):
    predicted_mental_health_burnout_score: float


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse("index.html")


@app.get("/")
def greet():
    return {"message": "Welcome to Manish Predictor Guys"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PersonData):
    input_row = pd.DataFrame([data.model_dump()])[list(model.feature_names_in_)]
    prediction = model.predict(input_row)[0]
    return PredictionResponse(
        predicted_mental_health_burnout_score=round(float(prediction), 2)
    )