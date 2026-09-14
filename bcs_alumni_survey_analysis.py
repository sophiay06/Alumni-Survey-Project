#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# config

LIKERT_SCORE = {
    "Not at all useful": 1,
    "Not useful": 2,
    "Somewhat useful": 3,
    "Useful": 4,
    "Very useful": 5,
    "Very Useful": 5,   # handles capitalization variation
}

REQUIRED_COURSE_CODES = [f"Q23_{i}" for i in range(1, 9)]
FOUNDATIONAL_COURSE_CODES = [f"Q37_{i}" for i in range(1, 12)]
ELECTIVE_COURSE_CODES = [f"Q24_{i}" for i in range(1, 52)]

OPEN_ENDED_CODES = [
    "Q19",         # disliked about BCS
    "Q29",         # should/should not be required
    "Q9",          # wanted classes not taken
    "Q25",         # useful outside courses
    "Q26_1_TEXT",  # missing class/topic
    "Q27",         # what would you change
    "Q38",         # favorite courses
    "Q28",         # anything else
    "Q21",         # current field/job/program
    "Q42",         # survey feedback
]

YES_NO_QUESTIONS = {
    "Q34": "Sufficient extracurricular BCS opportunities",
    "Q12": "Research opportunity felt accessible",
    "Q35": "Research should be required",
    "Q14": "Studied abroad",
}

MULTISELECT_QUESTIONS = {
    "Q17": "Essential BCS track themes",
    "Q40": "Would have taken more of",
    "Q20": "Current job/role",
    "Q13": "BCS opportunities outside classroom",
}

# Qualtrics stores some multi-select answers as one comma-separated string.
# A small number of choices themselves contain commas. Add known choices here
# so they are not accidentally split.
KNOWN_CHOICES_WITH_COMMAS = [
    "Research in the BCS department (e.g., working in a BCS lab, Research Assistant in a BCS lab)",
    "More natural science courses (physics, chemistry, engineering, etc)",
    "Student — Doctoral degree (PhD, MD, JD, PsyD, EdD, etc.)",
]


# cleaning

def read_qualtrics_csv(path: Path):
    """
    Qualtrics CSV exports contain:
      row 1: variable codes (Q1, Q2, ...)
      row 2: full question text
      row 3: ImportId metadata
      row 4+: actual responses

    We keep the full question text separately and load the responses using
    the variable codes as pandas column names.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f))

    if len(raw_rows) < 4:
        raise ValueError("The CSV does not look like a standard Qualtrics export.")

    codes = raw_rows[0]
    question_text = raw_rows[1]
    question_lookup = dict(zip(codes, question_text))

    # Read response rows only.
    df = pd.read_csv(
        path,
        header=0,
        skiprows=[1, 2],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    return df, question_lookup


def flag_probable_low_quality(row: pd.Series) -> bool:
    """
    Very conservative pilot-quality flag.

    It flags a response only when multiple open-text fields look like repeated
    keyboard-mashing strings such as 'asdf', rather than deleting it.
    """
    values = []
    for code in OPEN_ENDED_CODES:
        if code in row.index:
            text = str(row.get(code, "")).strip().lower()
            if text:
                values.append(text)

    suspicious = 0
    for text in values:
        collapsed = re.sub(r"\s+", "", text)
        if re.fullmatch(r"[asdfcjkl;]+", collapsed):
            suspicious += 1
        elif len(collapsed) >= 8 and len(set(collapsed)) <= 4:
            suspicious += 1

    return suspicious >= 3


def clean_sample(df: pd.DataFrame):
    """
    Primary analytic sample:
      - remove Survey Preview rows
      - keep respondents who answered Yes to BCS major
      - retain low-quality flags for transparency
    """
    non_preview = df[df["Status"].ne("Survey Preview")].copy()
    eligible = non_preview[non_preview["Q1"].eq("Yes")].copy()

    eligible["quality_flag"] = eligible.apply(
        lambda row: (
            "Probable low-quality/test-like"
            if flag_probable_low_quality(row)
            else "Usable"
        ),
        axis=1,
    )

    sensitivity = eligible[
        eligible["quality_flag"].eq("Usable")
    ].copy()

    return non_preview, eligible, sensitivity


# helpers

def extract_course_name(question: str) -> str:
    """Keep only the course label after the final ' - ' in a matrix question."""
    if not question:
        return ""
    return question.split(" - ")[-1].strip()


def split_multiselect(text: str):
    """
    Split Qualtrics multi-select text while protecting known response choices
    that contain commas.
    """
    text = str(text).strip()
    if not text:
        return []

    placeholders = {}
    protected = text

    for i, choice in enumerate(KNOWN_CHOICES_WITH_COMMAS):
        if choice in protected:
            token = f"__CHOICE_{i}__"
            placeholders[token] = choice
            protected = protected.replace(choice, token)

    pieces = [x.strip() for x in protected.split(",") if x.strip()]

    restored = []
    for piece in pieces:
        restored.append(placeholders.get(piece, piece))

    return restored


def save_table(df: pd.DataFrame, output_dir: Path, filename: str):
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


# analyses

def analyze_course_matrix(
    df: pd.DataFrame,
    codes: list[str],
    question_lookup: dict[str, str],
):
    """
    Analyze a course-usefulness matrix.

    'Did not take / do not remember' is excluded from the mean.
    """
    records = []

    for code in codes:
        if code not in df.columns:
            continue

        responses = df[code].astype(str).str.strip()
        numeric = responses.map(LIKERT_SCORE)
        valid = numeric.dropna()

        if len(valid) == 0:
            continue

        counts = responses.value_counts()

        records.append(
            {
                "course": extract_course_name(question_lookup.get(code, code)),
                "rated_n": int(valid.count()),
                "mean_usefulness_1_to_5": float(valid.mean()),
                "median_usefulness_1_to_5": float(valid.median()),
                "very_useful_n": int(
                    counts.get("Very useful", 0)
                    + counts.get("Very Useful", 0)
                ),
                "useful_n": int(counts.get("Useful", 0)),
                "somewhat_useful_n": int(counts.get("Somewhat useful", 0)),
                "not_useful_n": int(counts.get("Not useful", 0)),
                "not_at_all_useful_n": int(
                    counts.get("Not at all useful", 0)
                ),
                "did_not_take_or_remember_n": int(
                    responses.str.contains(
                        "Did not take|Do not remember",
                        case=False,
                        regex=True,
                    ).sum()
                ),
            }
        )

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            ["mean_usefulness_1_to_5", "rated_n"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return result


def analyze_yes_no(df: pd.DataFrame):
    records = []

    for code, label in YES_NO_QUESTIONS.items():
        if code not in df.columns:
            continue

        answers = df[code].astype(str).str.strip()
        yes = int((answers == "Yes").sum())
        no = int((answers == "No").sum())
        n = yes + no

        records.append(
            {
                "question": label,
                "yes_n": yes,
                "no_n": no,
                "valid_n": n,
                "yes_percent": (100 * yes / n) if n else None,
            }
        )

    return pd.DataFrame(records)


def analyze_multiselect(df: pd.DataFrame, code: str, label: str):
    counts = Counter()

    if code not in df.columns:
        return pd.DataFrame()

    for text in df[code]:
        for choice in split_multiselect(text):
            counts[choice] += 1

    n_respondents = len(df)

    records = [
        {
            "question": label,
            "choice": choice,
            "count": count,
            "percent_of_eligible_respondents": (
                100 * count / n_respondents if n_respondents else None
            ),
        }
        for choice, count in counts.most_common()
    ]

    return pd.DataFrame(records)


def respondent_profile(df: pd.DataFrame):
    records = []

    fields = {
        "Q2": "Graduation year",
        "Q3": "Degree type",
        "Q5": "Double major / dual degree",
        "Q6": "Minor",
    }

    for code, label in fields.items():
        if code not in df.columns:
            continue

        counts = df[code].astype(str).str.strip()
        counts = counts[counts.ne("")].value_counts()

        for category, count in counts.items():
            records.append(
                {
                    "measure": label,
                    "category": category,
                    "count": int(count),
                    "percent": 100 * count / len(df) if len(df) else None,
                }
            )

    return pd.DataFrame(records)


def numeric_rating_summary(df: pd.DataFrame):
    """
    Summarize single-number rating questions.
    """
    questions = {
        "Q8_1": "Overall rigor of BCS major",
        "Q16_1": "Ease of accommodating study abroad",
        "Q22_1": "BCS preparation for current job/role",
    }

    records = []

    for code, label in questions.items():
        if code not in df.columns:
            continue

        values = pd.to_numeric(df[code], errors="coerce").dropna()

        if values.empty:
            continue

        records.append(
            {
                "question": label,
                "n": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(records)


def export_open_ended(
    eligible: pd.DataFrame,
    sensitivity: pd.DataFrame,
    question_lookup: dict[str, str],
    output_dir: Path,
):
    """
    Export open-ended responses in long format so they are easy to hand-code.
    """

    def make_long(df, sample_name):
        rows = []

        for _, response in df.iterrows():
            response_id = response.get("ResponseId", "")

            for code in OPEN_ENDED_CODES:
                if code not in df.columns:
                    continue

                text = str(response.get(code, "")).strip()
                if not text:
                    continue

                rows.append(
                    {
                        "sample": sample_name,
                        "response_id": response_id,
                        "question_code": code,
                        "question": question_lookup.get(code, code),
                        "response": text,
                        "theme_code_1": "",
                        "theme_code_2": "",
                        "notes": "",
                    }
                )

        return rows

    all_rows = make_long(eligible, "Primary eligible sample")
    usable_rows = make_long(
        sensitivity,
        "Sensitivity sample excluding probable low-quality response",
    )

    long_df = pd.DataFrame(all_rows + usable_rows)
    save_table(long_df, output_dir, "open_ended_responses_for_coding.csv")
    return long_df


# figers

def plot_track_themes(track_df: pd.DataFrame, output_dir: Path):
    if track_df.empty:
        return

    plot_df = track_df.sort_values("count", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["choice"], plot_df["count"])
    plt.xlabel("Number of respondents")
    plt.ylabel("Track theme")
    plt.title("Essential BCS Track Themes")
    plt.tight_layout()
    plt.savefig(output_dir / "essential_track_themes.png", dpi=200)
    plt.close()


def plot_required_courses(required_df: pd.DataFrame, output_dir: Path):
    if required_df.empty:
        return

    plot_df = required_df.sort_values(
        "mean_usefulness_1_to_5",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        plot_df["course"],
        plot_df["mean_usefulness_1_to_5"],
    )
    plt.xlabel("Mean usefulness (1 = Not at all useful, 5 = Very useful)")
    plt.ylabel("Required course")
    plt.xlim(1, 5)
    plt.title("Required Course Usefulness")
    plt.tight_layout()
    plt.savefig(output_dir / "required_course_usefulness.png", dpi=200)
    plt.close()


def plot_yes_no(yes_no_df: pd.DataFrame, output_dir: Path):
    if yes_no_df.empty:
        return

    plot_df = yes_no_df.copy()

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["question"], plot_df["yes_percent"])
    plt.xlabel("Percent answering Yes")
    plt.xlim(0, 100)
    plt.title("BCS Research and Experience Questions")
    plt.tight_layout()
    plt.savefig(output_dir / "research_experience_yes_percent.png", dpi=200)
    plt.close()



def plot_required_courses_likert(df: pd.DataFrame, question_lookup: dict[str, str], output_dir: Path):
    """100% stacked horizontal bars for required-course usefulness."""
    categories = [
        "Not at all useful",
        "Not useful",
        "Somewhat useful",
        "Useful",
        "Very useful",
    ]

    rows = []
    for code in REQUIRED_COURSE_CODES:
        if code not in df.columns:
            continue

        responses = df[code].astype(str).str.strip().replace({"Very Useful": "Very useful"})
        valid = responses[responses.isin(categories)]

        if len(valid) == 0:
            continue

        counts = valid.value_counts()
        total = len(valid)

        row = {
            "course": extract_course_name(question_lookup.get(code, code)),
            "rated_n": total,
        }
        for cat in categories:
            row[cat] = 100 * counts.get(cat, 0) / total

        rows.append(row)

    if not rows:
        return

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 7))
    left = [0] * len(plot_df)

    for cat in categories:
        vals = plot_df[cat].tolist()
        plt.barh(plot_df["course"], vals, left=left, label=cat)
        left = [l + v for l, v in zip(left, vals)]

    plt.xlabel("Percent of respondents who rated the course")
    plt.ylabel("Required course")
    plt.xlim(0, 100)
    plt.title("Required Course Usefulness — Response Distribution")
    plt.legend(title="Usefulness", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "required_course_usefulness_likert_stacked.png", dpi=200)
    plt.close()


def plot_rigor_distribution(df: pd.DataFrame, output_dir: Path):
    """Distribution of the intentional 0–5 BCS rigor rating."""
    if "Q8_1" not in df.columns:
        return

    values = pd.to_numeric(df["Q8_1"], errors="coerce").dropna()
    counts = values.value_counts().reindex(range(0, 6), fill_value=0)

    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in range(0, 6)], counts.values)
    plt.xlabel("BCS rigor rating (0–5)")
    plt.ylabel("Number of respondents")
    plt.title("Distribution of BCS Rigor Ratings")
    plt.tight_layout()
    plt.savefig(output_dir / "bcs_rigor_distribution.png", dpi=200)
    plt.close()


def plot_yes_no_100_stacked(yes_no_df: pd.DataFrame, output_dir: Path):
    """100% stacked Yes/No bars for binary questions."""
    if yes_no_df.empty:
        return

    plot_df = yes_no_df.copy()
    plot_df["no_percent"] = 100 - plot_df["yes_percent"]

    plt.figure(figsize=(10, 5.5))
    plt.barh(plot_df["question"], plot_df["yes_percent"], label="Yes")
    plt.barh(
        plot_df["question"],
        plot_df["no_percent"],
        left=plot_df["yes_percent"],
        label="No",
    )
    plt.xlabel("Percent of valid responses")
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.title("Research and Experience Questions — Yes/No Distribution")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "research_experience_yes_no_100_stacked.png", dpi=200)
    plt.close()


def plot_missingness_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Pilot QA heatmap.
    1 = response present; 0 = blank/not shown.
    """
    selected = [
        ("Q1", "BCS major?"),
        ("Q2", "Graduation year"),
        ("Q3", "Degree type"),
        ("Q8_1", "Rigor 0–5"),
        ("Q12", "Research accessible?"),
        ("Q14", "Studied abroad?"),
        ("Q16_1", "Study abroad follow-up"),
        ("Q17", "Essential tracks"),
        ("Q22_1", "Preparation rating"),
        ("Q27", "Program change"),
        ("Q28", "Other comments"),
        ("Q42", "Survey feedback"),
    ]

    present = []
    labels = []

    for code, label in selected:
        if code in df.columns:
            vals = df[code].astype(str).str.strip()
            present.append((vals != "").astype(int).tolist())
            labels.append(label)

    if not present:
        return

    matrix = pd.DataFrame(present, index=labels).T

    plt.figure(figsize=(11, 5.5))
    plt.imshow(matrix.T.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    plt.xticks(range(len(matrix)), [f"R{i+1}" for i in range(len(matrix))])
    plt.xlabel("Eligible pilot respondent")
    plt.ylabel("Survey item")
    plt.title("Pilot Response Completion / Conditional-Logic Check")
    plt.colorbar(label="0 = blank, 1 = response present")
    plt.tight_layout()
    plt.savefig(output_dir / "pilot_missingness_conditional_logic_heatmap.png", dpi=200)
    plt.close()



#main

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the BCS Curriculum Alumni Qualtrics survey."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="BCS Curriculum Pilot Survey_September 13, 2026_14.15.csv",
        help="Path to the Qualtrics CSV export.",
    )
    parser.add_argument(
        "--output",
        default="bcs_survey_analysis_output",
        help="Directory where analysis results will be saved.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n  {csv_path}\n\n"
            "Put the .py file in the same folder as the CSV, or pass the "
            "full CSV path on the command line."
        )

    # ---------------- LOAD ----------------
    df, question_lookup = read_qualtrics_csv(csv_path)

    # ---------------- CLEAN ----------------
    non_preview, eligible, sensitivity = clean_sample(df)

    # Save cleaned respondent-level data.
    save_table(
        eligible,
        output_dir,
        "cleaned_eligible_responses.csv",
    )

    # ---------------- SAMPLE SUMMARY ----------------
    sample_summary = pd.DataFrame(
        [
            {
                "metric": "Total response rows",
                "value": len(df),
            },
            {
                "metric": "Survey Preview rows removed",
                "value": int((df["Status"] == "Survey Preview").sum()),
            },
            {
                "metric": "Non-preview responses",
                "value": len(non_preview),
            },
            {
                "metric": "Eligible BCS alumni respondents",
                "value": len(eligible),
            },
            {
                "metric": "Probable low-quality/test-like responses",
                "value": int(
                    (
                        eligible["quality_flag"]
                        == "Probable low-quality/test-like"
                    ).sum()
                ),
            },
            {
                "metric": "Sensitivity sample size",
                "value": len(sensitivity),
            },
        ]
    )
    save_table(sample_summary, output_dir, "sample_summary.csv")

    # ---------------- PROFILE ----------------
    profile = respondent_profile(eligible)
    save_table(profile, output_dir, "respondent_profile.csv")

    # ---------------- RATINGS ----------------
    rating_summary = numeric_rating_summary(eligible)
    save_table(
        rating_summary,
        output_dir,
        "numeric_rating_summary.csv",
    )

    # ---------------- YES / NO ----------------
    yes_no = analyze_yes_no(eligible)
    save_table(yes_no, output_dir, "yes_no_summary.csv")

    # ---------------- MULTI-SELECT ----------------
    multiselect_results = {}

    for code, label in MULTISELECT_QUESTIONS.items():
        table = analyze_multiselect(eligible, code, label)
        multiselect_results[code] = table

        filename = (
            re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            + ".csv"
        )
        save_table(table, output_dir, filename)

    # ---------------- COURSE USEFULNESS ----------------
    required = analyze_course_matrix(
        eligible,
        REQUIRED_COURSE_CODES,
        question_lookup,
    )
    foundational = analyze_course_matrix(
        eligible,
        FOUNDATIONAL_COURSE_CODES,
        question_lookup,
    )
    electives = analyze_course_matrix(
        eligible,
        ELECTIVE_COURSE_CODES,
        question_lookup,
    )

    save_table(
        required,
        output_dir,
        "required_course_usefulness.csv",
    )
    save_table(
        foundational,
        output_dir,
        "foundational_course_usefulness.csv",
    )
    save_table(
        electives,
        output_dir,
        "elective_course_usefulness.csv",
    )

    # ---------------- OPEN-ENDED ----------------
    export_open_ended(
        eligible,
        sensitivity,
        question_lookup,
        output_dir,
    )

    # ---------------- FIGURES ----------------
    track_df = multiselect_results.get("Q17", pd.DataFrame())
    plot_track_themes(track_df, output_dir)
    plot_required_courses(required, output_dir)
    plot_yes_no(yes_no, output_dir)

    # Additional presentation-friendly pilot QA / visualization examples
    plot_required_courses_likert(eligible, question_lookup, output_dir)
    plot_rigor_distribution(eligible, output_dir)
    plot_yes_no_100_stacked(yes_no, output_dir)
    plot_missingness_heatmap(eligible, output_dir)

    # ---------------- PRINT RESULTS ----------------
    print("\n" + "=" * 72)
    print("BCS CURRICULUM ALUMNI SURVEY — PILOT ANALYSIS")
    print("=" * 72)

    print("\nSAMPLE CLEANING")
    print(sample_summary.to_string(index=False))

    if not rating_summary.empty:
        print("\nNUMERIC RATINGS")
        print(
            rating_summary.round(2).to_string(index=False)
        )

    if not yes_no.empty:
        print("\nYES / NO QUESTIONS")
        print(
            yes_no.round(
                {"yes_percent": 1}
            ).to_string(index=False)
        )

    if not track_df.empty:
        print("\nESSENTIAL TRACK THEMES")
        print(
            track_df[
                [
                    "choice",
                    "count",
                    "percent_of_eligible_respondents",
                ]
            ]
            .round({"percent_of_eligible_respondents": 1})
            .to_string(index=False)
        )

    if not required.empty:
        print("\nREQUIRED COURSE USEFULNESS")
        print(
            required[
                [
                    "course",
                    "rated_n",
                    "mean_usefulness_1_to_5",
                ]
            ]
            .round({"mean_usefulness_1_to_5": 2})
            .to_string(index=False)
        )

    print("\n" + "-" * 72)
    print("Files saved to:")
    print(output_dir.resolve())
    print("-" * 72)
    print(
        "\nInterpretation note: This is pilot data with a very small sample. "
        "Use descriptive statistics and qualitative themes rather than "
        "significance tests."
    )


if __name__ == "__main__":
    main()
