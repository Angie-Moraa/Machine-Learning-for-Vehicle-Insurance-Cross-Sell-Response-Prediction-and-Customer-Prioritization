# AI Use Disclosure

**Student:** Angela Moraa Omweri (223763) · DSA 8401 Applied Machine Learning

## Tool used
Claude (Anthropic), used as an assistant through the Claude app in November 2026.

## What I did myself
- Chose the problem and dataset, and wrote and submitted the proposal. 
- Wrote and followed up with the EDA section based on knowledge i have.
- Made the modelling decisions listed in the EDA (temporal split, target encoding, PR-AUC, agent and fairness checks).
- Ran every notebook, script, pipeline step, the API and the Docker build on my own machine, and checked the outputs.
- Set up the GitHub repository, Docker Desktop, MLflow and the Render deployment.
- Model training - drafting the classification notebook and writing explanations of the results I produced.
-Interpretation - explaining the SHAP chart, MLflow and the fairness results.

## What AI assisted with
- **Structure:** showed me parts that i may have not covered that were neede from the school notes.
- **Documentation:** drafting `README.md` and the final report, which I reviewed and edited.
- **Troubleshooting:** installing packages, Git commands, Docker and WSL setup, and Render deployment.

## How I verified the AI's output
- Every number in the report and README comes from outputs I ran (notebook results, `artifacts/model_report.json`, logs).
- The pipeline results were checked against the notebook (for example, test PR-AUC 0.362 against 0.365).
- The live API was tested with the form and `/health`.

## Responsibility
I understand all the code and decisions in this repository and can explain and defend them.