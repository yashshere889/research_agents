from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    key_contributions: list[str] = Field(default_factory=list)
    methodology_notes: str = ""


class LiteratureReviewOutput(BaseModel):
    papers_retrieved: list[ArxivPaper]
    research_gaps: list[str]
    existing_methods: list[str]
    summary: str
    suggested_directions: list[str]


class Milestone(BaseModel):
    title: str
    description: str
    deliverable: str
    estimated_hours: int = Field(ge=0)
    dependencies: list[str] = Field(default_factory=list)


class ResearchPlanOutput(BaseModel):
    hypothesis: str
    objectives: list[str]
    methodology: str
    milestones: list[Milestone]
    required_datasets: list[str]
    required_libraries: list[str]
    evaluation_metrics: list[str]
    risks_and_mitigations: list[str]


class Dataset(BaseModel):
    name: str
    source: str
    local_path: str
    num_samples: int = Field(ge=0)
    features: list[str]
    preprocessing_steps: list[str]
    train_test_split: dict[str, int]


class DataPreparationOutput(BaseModel):
    datasets: list[Dataset]
    preprocessing_code: str
    data_statistics: dict[str, Any]
    feature_engineering_notes: str
    data_quality_report: str


class ExperimentRun(BaseModel):
    experiment_id: str
    description: str
    code_executed: str
    stdout: str
    stderr: str
    metrics: dict[str, float]
    artifacts: list[str]
    status: str


class ExperimentOutput(BaseModel):
    runs: list[ExperimentRun]
    best_run_id: str
    conclusions: list[str]
    ablation_notes: str
    reproducibility_notes: str


class PaperSection(BaseModel):
    title: str
    content: str


class PaperOutput(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
    sections: list[PaperSection]
    references: list[str]
    full_markdown: str
    pdf_path: str | None = None

