from __future__ import annotations

import json
import uuid

from research_agents.agents.base_agent import BaseAgent
from research_agents.agents.experiment_runner.prompts import SYSTEM_PROMPT
from research_agents.agents.experiment_runner.sandbox import SandboxSession
from research_agents.agents.experiment_runner.tools import parse_metrics
from research_agents.models.agent_outputs import ExperimentOutput, ExperimentRun


class ExperimentRunnerAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def run(self, state) -> ExperimentOutput:
        if not state.research_plan or not state.data_preparation:
            raise ValueError("Research plan and data preparation are required.")
        plan = state.research_plan
        data = state.data_preparation
        scripts = self._design_experiments(plan, data)
        runs: list[ExperimentRun] = []

        with SandboxSession() as sandbox:
            sandbox.upload_data(data)
            for experiment in scripts:
                result = sandbox.run_code(experiment["code"])
                stdout = getattr(result, "stdout", "") or ""
                stderr = getattr(result, "stderr", "") or ""
                is_error = bool(getattr(result, "error", False) or stderr)
                runs.append(
                    ExperimentRun(
                        experiment_id=str(uuid.uuid4())[:8],
                        description=experiment["description"],
                        code_executed=experiment["code"],
                        stdout=stdout,
                        stderr=stderr,
                        metrics=parse_metrics(stdout),
                        artifacts=sandbox.list_artifacts(),
                        status="failed" if is_error else "success",
                    )
                )

        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Experiment runs:\n{[run.model_dump() for run in runs]}\n"
                        f"Research hypothesis: {plan.hypothesis}\n"
                        f"Expected metrics: {plan.evaluation_metrics}\n"
                        "Synthesise into ExperimentOutput JSON."
                    ),
                }
            ],
            response_model=ExperimentOutput,
        )

    def _design_experiments(self, plan, data) -> list[dict[str, str]]:
        response = self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"""Design experiments to test: {plan.hypothesis}
Methodology: {plan.methodology}
Available data: {[dataset.name for dataset in data.datasets]}
Evaluation metrics: {plan.evaluation_metrics}

Return a JSON list of {{"description": str, "code": str}} objects.
Each code block must load data from ./data/processed/, train/evaluate, and print metrics as JSON.""",
                }
            ]
        )
        experiments = json.loads(response)
        if not isinstance(experiments, list):
            raise ValueError("Experiment design must return a JSON list.")
        return experiments

