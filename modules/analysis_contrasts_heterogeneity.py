import os

import numpy as np
import pandas


def quote_formula_col(col: str) -> str:
    return 'Q("%s")' % col


def categorical_formula_col(col: str) -> str:
    return "C(%s)" % quote_formula_col(col)


def categorical_level_term(col: str, level: object) -> str:
    """
    Patsy/statsmodels coefficient name for a non-reference level
    of a categorical variable.

    Example:
    C(Q("selected_likelihood_rating"))[T.2]
    """
    return "%s[T.%s]" % (
        categorical_formula_col(col),
        level,
    )


def interaction_term(term_1: str, term_2: str) -> str:
    return "%s:%s" % (term_1, term_2)


def significance_stars(p_value: float) -> str:
    if pandas.isna(p_value):
        return ""

    if p_value < 0.0001:
        return "****"

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return ""


def get_reference_level(
    df: pandas.DataFrame,
    col: str,
) -> object:
    """
    Patsy's default treatment coding uses one category as the omitted
    reference level.

    For ordinary string/numeric columns, the categories are generally
    ordered according to Patsy's category ordering.

    This function reproduces the common case by sorting the observed
    levels.
    """
    values = df[col].dropna().unique().tolist()

    try:
        return sorted(values)[0]
    except TypeError:
        return sorted(values, key=str)[0]


def get_nonreference_levels(
    df: pandas.DataFrame,
    col: str,
) -> list[object]:
    reference = get_reference_level(df, col)

    values = df[col].dropna().unique().tolist()

    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)

    return [
        value
        for value in values
        if value != reference
    ]


def build_contrast_vector(
    model: object,
    terms_to_add: list[str],
) -> np.ndarray:
    """
    Construct a linear contrast vector.

    If the desired implied effect is:

        beta_rating2
        + beta_rating2_x_hobbies

    this creates a vector with 1 for those two coefficients and
    0 for every other coefficient.
    """
    param_names = list(model.params.index)

    contrast = np.zeros(len(param_names))

    for term in terms_to_add:
        if term not in param_names:
            raise KeyError(
                "Could not find coefficient term in fitted model:\n"
                "%s\n\nAvailable terms:\n%s"
                % (
                    term,
                    "\n".join(param_names),
                )
            )

        term_index = param_names.index(term)
        contrast[term_index] = 1.0

    return contrast


def run_linear_contrast(
    model: object,
    terms_to_add: list[str],
) -> dict[str, float]:
    """
    Test:

        sum(terms_to_add) = 0

    using the covariance matrix associated with the fitted model.

    Because the original model was fit with cluster-robust covariance,
    this test uses the corresponding robust covariance information.
    """
    contrast_vector = build_contrast_vector(
        model,
        terms_to_add,
    )

    test_result = model.t_test(contrast_vector)

    estimate = float(np.asarray(test_result.effect).reshape(-1)[0])
    std_error = float(np.asarray(test_result.sd).reshape(-1)[0])
    statistic = float(np.asarray(test_result.tvalue).reshape(-1)[0])
    p_value = float(np.asarray(test_result.pvalue).reshape(-1)[0])

    conf_int = np.asarray(test_result.conf_int(alpha=0.05))
    ci_low = float(conf_int.reshape(-1, 2)[0, 0])
    ci_high = float(conf_int.reshape(-1, 2)[0, 1])

    return {
        "estimate": estimate,
        "std_error": std_error,
        "statistic": statistic,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significance": significance_stars(p_value),
    }


def build_categorical_implied_effects_for_model(
    model_name: str,
    model: object,
    analysis_df: pandas.DataFrame,
    cfg: object,
) -> pandas.DataFrame:
    """
    Calculate implied likelihood-rating effects for every moderator level.

    Example for Hobbies and rating 2:

        rating_2 coefficient
        +
        rating_2 x Hobbies interaction

    If multiple judgment models are present and the original specification
    contains rating x judgment_model interactions, the appropriate
    rating x judgment_model term is also added.

    There is intentionally NO moderator x judgment_model term here because
    the current heterogeneity model does not fit a three-way interaction.
    """
    rating_col = cfg.analysis.selected_rating_col
    moderator_col = cfg.analysis.moderator_col
    judgment_model_col = cfg.analysis.judgment_model_col

    rating_reference = get_reference_level(
        analysis_df,
        rating_col,
    )
    moderator_reference = get_reference_level(
        analysis_df,
        moderator_col,
    )
    judgment_model_reference = get_reference_level(
        analysis_df,
        judgment_model_col,
    )

    rating_levels = get_nonreference_levels(
        analysis_df,
        rating_col,
    )

    moderator_levels = analysis_df[
        moderator_col
    ].dropna().unique().tolist()

    judgment_models = analysis_df[
        judgment_model_col
    ].dropna().unique().tolist()

    try:
        moderator_levels = sorted(moderator_levels)
    except TypeError:
        moderator_levels = sorted(
            moderator_levels,
            key=str,
        )

    try:
        judgment_models = sorted(judgment_models)
    except TypeError:
        judgment_models = sorted(
            judgment_models,
            key=str,
        )

    rows = []

    for judgment_model in judgment_models:
        for moderator_level in moderator_levels:
            for rating_level in rating_levels:

                rating_term = categorical_level_term(
                    rating_col,
                    rating_level,
                )

                terms_to_add = [
                    rating_term,
                ]

                # -------------------------------------------------
                # Moderator interaction
                #
                # For the reference moderator, there is no interaction
                # coefficient to add.
                # -------------------------------------------------
                if moderator_level != moderator_reference:
                    moderator_term = categorical_level_term(
                        moderator_col,
                        moderator_level,
                    )

                    rating_moderator_interaction = interaction_term(
                        rating_term,
                        moderator_term,
                    )

                    terms_to_add.append(
                        rating_moderator_interaction
                    )

                # -------------------------------------------------
                # Judgment-model interaction
                #
                # Your heterogeneity formula preserves the original
                # rating x judgment_model interaction.
                #
                # If only one judgment model is in the dataset,
                # nothing is added here.
                # -------------------------------------------------
                if (
                    cfg.analysis.include_model_interactions
                    and judgment_model != judgment_model_reference
                ):
                    judgment_model_term = categorical_level_term(
                        judgment_model_col,
                        judgment_model,
                    )

                    rating_model_interaction = interaction_term(
                        rating_term,
                        judgment_model_term,
                    )

                    terms_to_add.append(
                        rating_model_interaction
                    )

                result = run_linear_contrast(
                    model,
                    terms_to_add,
                )

                rows.append(
                    {
                        "model": model_name,
                        "judgment_model": judgment_model,
                        "moderator": moderator_col,
                        "moderator_level": moderator_level,
                        "moderator_reference": moderator_reference,
                        "rating_reference": rating_reference,
                        "rating_level": rating_level,
                        "contrast": "%s vs %s"
                        % (
                            rating_level,
                            rating_reference,
                        ),
                        "terms_added": " + ".join(
                            terms_to_add
                        ),
                        **result,
                    }
                )

    return pandas.DataFrame(rows)


def build_all_categorical_implied_effects(
    models: dict[str, object],
    sample_dfs: dict[str, pandas.DataFrame],
    cfg: object,
) -> pandas.DataFrame:
    """
    Generate post-estimation contrasts for the categorical heterogeneity
    models.

    This keeps:
        categorical_lpm
        categorical_linear

    and skips:
        ordinal_lpm
        ordinal_linear
        logits
    """
    result_dfs = []

    for model_name, model in models.items():

        if "__categorical_lpm_" not in model_name and \
           "__categorical_linear_" not in model_name:
            continue

        sample_name = model_name.split("__")[0]

        analysis_df = sample_dfs[sample_name]

        result_df = build_categorical_implied_effects_for_model(
            model_name=model_name,
            model=model,
            analysis_df=analysis_df,
            cfg=cfg,
        )

        result_df["sample"] = sample_name

        result_dfs.append(result_df)

    if not result_dfs:
        return pandas.DataFrame()

    return pandas.concat(
        result_dfs,
        ignore_index=True,
    )


def write_categorical_implied_effects(
    contrast_df: pandas.DataFrame,
    output_dir: str,
) -> None:
    output_path = os.path.join(
        output_dir,
        "heterogeneity_implied_effects.csv",
    )

    contrast_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "Wrote post-estimation implied effects to: %s"
        % output_path
    )