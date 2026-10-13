import os
import pandas
from modules.console import print_subbanner

def model_to_table(model_name: str, model: object) -> pandas.DataFrame:
    conf_int = pandas.DataFrame(
        model.conf_int(),
        index=model.params.index,
    )
    table = pandas.DataFrame(
        {
            "model": model_name,
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "statistic": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": conf_int.iloc[:, 0].values,
            "ci_high": conf_int.iloc[:, 1].values,
            "nobs": model.nobs,
        }
    )
    return table


def models_to_table(models: dict[str, object]) -> pandas.DataFrame:
    tables = []
    for model_name, model in models.items():
        tables.append(model_to_table(model_name, model))
    return pandas.concat(tables, ignore_index=True)


def write_model_outputs(
    models: dict[str, object],
    table_df: pandas.DataFrame,
    output_dir: str
) -> None:
    table_path = os.path.join(output_dir, "regression_tables.csv")
    markdown_path = os.path.join(output_dir, "regression_tables.md")
    summary_path = os.path.join(output_dir, "regression_summaries.txt")
    stargazer_tables = build_stargazer_tables(models)

    table_df.to_csv(table_path, index=False)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(dataframe_to_markdown(table_df))

    write_stargazer_outputs(stargazer_tables, output_dir)

    with open(summary_path, "w", encoding="utf-8") as file:
        for model_name, model in models.items():
            file.write("%s\n" % model_name)
            file.write("%s\n\n" % model.summary().as_text())


def print_model_outputs(models: dict[str, object], table_df: pandas.DataFrame) -> None:
    print_subbanner("Coefficient Table")
    print(dataframe_to_markdown(table_df))
    for table_name, stargazer_df in build_stargazer_tables(models).items():
        print_subbanner("Stargazer-Style Table: %s" % table_name)
        print(stargazer_to_text(stargazer_df))
    for model_name, model in models.items():
        print_subbanner("Model Summary: %s" % model_name)
        print(model.summary())


def write_stargazer_outputs(
    stargazer_tables: dict[str, pandas.DataFrame],
    output_dir: str
) -> None:
    for table_name, stargazer_df in stargazer_tables.items():
        csv_path = os.path.join(
            output_dir,
            "regression_stargazer_%s.csv" % table_name,
        )
        markdown_path = os.path.join(
            output_dir,
            "regression_stargazer_%s.md" % table_name,
        )
        text_path = os.path.join(
            output_dir,
            "regression_stargazer_%s.txt" % table_name,
        )
        tex_path = os.path.join(
            output_dir,
            "regression_stargazer_%s.tex" % table_name,
        )
        stargazer_df.to_csv(csv_path, index=False)
        with open(markdown_path, "w", encoding="utf-8") as file:
            file.write(stargazer_to_markdown(stargazer_df))
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(stargazer_to_text(stargazer_df))
        with open(tex_path, "w", encoding="utf-8") as file:
            file.write(stargazer_to_latex(stargazer_df))


def build_stargazer_tables(models: dict[str, object]) -> dict[str, pandas.DataFrame]:
    grouped_models = {}
    for model_name, model in models.items():
        table_name = stargazer_table_name(model_name)
        if table_name not in grouped_models:
            grouped_models[table_name] = {}
        grouped_models[table_name][model_name] = model

    stargazer_tables = {}
    for table_name, table_models in grouped_models.items():
        stargazer_tables[table_name] = models_to_stargazer_table(table_models)

    return stargazer_tables


def stargazer_table_name(model_name: str) -> str:
    parts = model_name.split("__")
    model_parts = parts[1].split("_")
    spec_name = model_parts[0]
    family_name = model_parts[1]
    table_family = grouped_family_name(family_name)
    return "%s_%s" % (
        spec_name,
        table_family,
    )


def grouped_family_name(family_name: str) -> str:
    if family_name == "logit":
        return "logit"

    return "linear"


def models_to_stargazer_table(models: dict[str, object]) -> pandas.DataFrame:
    terms = collect_model_terms(models)
    rows = []
    for term in terms:
        estimate_row = {
            "term": clean_term(term),
        }
        se_row = {
            "term": "",
        }
        for model_name, model in models.items():
            if term in model.params.index:
                estimate_row[model_name] = format_estimate(
                    model.params[term],
                    model.pvalues[term],
                )
                se_row[model_name] = "(%s)" % format_float(model.bse[term])
            else:
                estimate_row[model_name] = ""
                se_row[model_name] = ""
        rows.append(estimate_row)
        rows.append(se_row)

    nobs_row = {
        "term": "Observations",
    }
    for model_name, model in models.items():
        nobs_row[model_name] = "%s" % int(model.nobs)
    rows.append(nobs_row)

    return pandas.DataFrame(rows)


def collect_model_terms(models: dict[str, object]) -> list[str]:
    terms = []
    for _, model in models.items():
        for term in model.params.index:
            if term not in terms:
                terms.append(term)
    return terms


def format_estimate(estimate: float, p_value: float) -> str:
    return "%s%s" % (
        format_float(estimate),
        significance_stars(p_value),
    )


def format_float(value: float) -> str:
    return "%.4f" % value


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"

    if p_value < 0.05:
        return "**"

    if p_value < 0.1:
        return "*"

    return ""


def clean_term(term: str) -> str:
    cleaned = term.replace("Q(\"", "")
    cleaned = cleaned.replace("\")", "")
    cleaned = cleaned.replace("C(", "")
    cleaned = cleaned.replace(")[T.", " = ")
    cleaned = cleaned.replace("]", "")
    return cleaned


def stargazer_to_markdown(df: pandas.DataFrame) -> str:
    return dataframe_to_markdown(df, format_numeric=False)


def stargazer_to_text(df: pandas.DataFrame) -> str:
    widths = {}
    for col in df.columns:
        values = [str(value) for value in df[col].tolist()]
        widths[col] = max([len(col)] + [len(value) for value in values])

    header = "  ".join([col.ljust(widths[col]) for col in df.columns])
    divider = "  ".join(["-" * widths[col] for col in df.columns])
    lines = [
        header,
        divider,
    ]
    for _, row in df.iterrows():
        lines.append(
            "  ".join([str(row[col]).ljust(widths[col]) for col in df.columns])
        )

    lines.append("")
    lines.append("* p < 0.1, ** p < 0.05, *** p < 0.01")
    return "\n".join(lines) + "\n"


def stargazer_to_latex(df: pandas.DataFrame) -> str:
    columns = list(df.columns)
    column_spec = "l" + "c" * (len(columns) - 1)
    lines = [
        "\\begin{tabular}{%s}" % column_spec,
        "\\hline",
        " & ".join([latex_escape(col) for col in columns]) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join([latex_escape(str(row[col])) for col in columns]) + " \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\multicolumn{%s}{l}{* p < 0.1, ** p < 0.05, *** p < 0.01} \\\\" % len(columns),
            "\\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def dataframe_to_markdown(
    df: pandas.DataFrame,
    format_numeric: bool = True
) -> str:
    text_df = df.copy()
    if format_numeric:
        for col in [
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "ci_low",
            "ci_high",
        ]:
            text_df[col] = text_df[col].map(lambda value: "%.4f" % value)

    headers = list(text_df.columns)
    lines = [
        "| %s |" % " | ".join(headers),
        "| %s |" % " | ".join(["---"] * len(headers)),
    ]
    for _, row in text_df.iterrows():
        lines.append("| %s |" % " | ".join([str(row[col]) for col in headers]))

    return "\n".join(lines) + "\n"
