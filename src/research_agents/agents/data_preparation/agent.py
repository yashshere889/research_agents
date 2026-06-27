from __future__ import annotations

from research_agents.agents.base_agent import BaseAgent
from research_agents.agents.data_preparation.prompts import SYSTEM_PROMPT
from research_agents.agents.experiment_runner.sandbox import SandboxSession
from research_agents.models.agent_outputs import DataPreparationOutput


class DataPreparationAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def run(self, state) -> DataPreparationOutput:
        if not state.research_plan:
            raise ValueError("Research plan must exist before data preparation.")
        plan = state.research_plan
        prep_plan = self._plan_data_pipeline(plan)
        code = self._generate_preprocessing_code(prep_plan, plan)

        with SandboxSession() as sandbox:
            result = sandbox.run_code(code)

        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"""Data pipeline plan:
{prep_plan}

Code executed:
```python
{code}
```

Execution output:
{getattr(result, "stdout", "")}

Errors:
{getattr(result, "stderr", "")}

Parse results into a DataPreparationOutput JSON.""",
                }
            ],
            response_model=DataPreparationOutput,
        )

    def _plan_data_pipeline(self, plan) -> str:
        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Required datasets: {plan.required_datasets}\n"
                        f"Libraries: {plan.required_libraries}\n"
                        "Describe a concrete data acquisition and preprocessing pipeline."
                    ),
                }
            ]
        )

    def _generate_preprocessing_code(self, pipeline_plan: str, plan) -> str:
        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"""Write complete, runnable Python code to:
1. Download/load the datasets: {plan.required_datasets}
2. Preprocess according to: {pipeline_plan}
3. Output statistics and save processed data to ./data/processed/

Include error handling, seed=42, type hints, and brief inline comments for non-obvious logic.
Return ONLY the Python code, no markdown fences.""",
                }
            ]
        )

