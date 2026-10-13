import pandas

REQUIRED_COLUMNS = [
    "base_experiment_ID",
    "detail",
    "option_1",
    "option_2",
    "forward_prompt",
    "forward_swapped_prompt",
    "forward_answer_potential_statement1",
    "forward_answer_potential_statement2",
    "reverse_option_1",
    "reverse_option_2",
    "reverse_potential_prompt1",
    "reverse_potential_prompt2",
    "reverse_potential_swapped_prompt1",
    "reverse_potential_swapped_prompt2",
]


def load_dataset(dataset_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(dataset_path)
    return validate_input_dataset(df)


def validate_input_dataset(df: pandas.DataFrame) -> pandas.DataFrame:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: %s" % missing_cols)

    duplicate_mask = df["base_experiment_ID"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            [
                "base_experiment_ID",
                "option_1",
                "option_2",
            ],
        ].sort_values("base_experiment_ID")

        raise ValueError(
            "Duplicate base_experiment_ID values found:\n%s"
            % duplicate_rows.to_string(index=False)
        )

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            missing_ids = df.loc[df[col].isna(), "base_experiment_ID"].tolist()
            raise ValueError(
                "Missing values found in column \"%s\" for IDs: %s"
                % (
                    col,
                    missing_ids,
                )
            )

    same_options_mask = (
        df["option_1"].astype(str).str.lower().str.strip()
        == df["option_2"].astype(str).str.lower().str.strip()
    )
    if same_options_mask.any():
        bad_rows = df.loc[
            same_options_mask,
            [
                "base_experiment_ID",
                "option_1",
                "option_2",
            ],
        ]
        raise ValueError(
            "Some rows have identical option_1 and option_2:\n%s"
            % bad_rows.to_string(index=False)
        )

    same_reverse_mask = (
        df["reverse_option_1"].astype(str).str.lower().str.strip()
        == df["reverse_option_2"].astype(str).str.lower().str.strip()
    )
    if same_reverse_mask.any():
        bad_rows = df.loc[
            same_reverse_mask,
            [
                "base_experiment_ID",
                "reverse_option_1",
                "reverse_option_2",
            ],
        ]
        raise ValueError(
            "Some rows have identical reverse_option_1 and reverse_option_2:\n%s"
            % bad_rows.to_string(index=False)
        )

    print("Dataset validation passed.")
    return df
