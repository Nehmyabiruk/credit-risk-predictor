FEATURE_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

ENGINEERED_COLUMNS = [
    "income_log",
    "income_log_capped",
    "debt_income",
    "age_ratio",
    "loan_history",
]

TARGET = "loan_status"

CATEGORICAL = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "income_log",
    "income_log_capped",
    "debt_income",
    "age_ratio",
    "loan_history",
]