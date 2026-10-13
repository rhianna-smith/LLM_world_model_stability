# LLM World Model Stability

Run one judgment model:

```bash
python run_experiment.py model=gpt4_1
python run_experiment.py model=gpt5_4_mini
python run_experiment.py model=llama3_8B
python run_experiment.py model=gemma_3_4B
```

Run event plausibility annotations:

```bash
python run_event_plausibility_annotations.py model=gemini3_1_flash_lite
```

Run regression analysis:

```bash
python run_analysis.py
```

The experiment is configured through Hydra:

- `configs/config.yaml` sets dataset and output roots.
- `configs/analysis.yaml` sets judgment result paths, rating paths, output directory, selected-option plausibility mapping, model variable, ordinal and categorical predictors, samples, outcomes, and clustering column.
- `configs/event_plausibility_annotations.yaml` sets the Gemini-only annotation dataset, output directory, and artifact filenames.
- `configs/model/*.yaml` sets the LiteLLM model name, output directory name, retry count, and generation settings.
- `modules/schemas.py` defines the structured LLM response schemas and plausibility-rating enum used by LiteLLM.
- `run.max_workers` controls script-level thread parallelism for LLM calls.
- Each run saves `resolved_config.yaml` into the model output directory.



The paths specified in the scripts assume that the dataset is located in the data directory. The default paths are configured for GPT-5.4-mini.

Run the following commands in order:

```bash
python run_event_plausibility_annotations.py model=gemini3_1_flash_lite
python run_experiment.py model=gpt5_4_mini
python run_analysis.py
```
