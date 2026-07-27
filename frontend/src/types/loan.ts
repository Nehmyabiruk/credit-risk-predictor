export interface LoanData {
  person_age: number;
  person_gender: string;
  person_education: string;
  person_income: number;
  person_emp_exp: number;
  person_home_ownership: string;
  loan_amnt: number;
  loan_intent: string;
  loan_int_rate: number;
  loan_percent_income: number;
  cb_person_cred_hist_length: number;
  credit_score: number;
  previous_loan_defaults_on_file: string;
}

export interface PredictionResponse {
  id: number;
  prediction: number;
  default_probability: number;
  risk: string;
  explanations: FeatureExplanation[];
}

export interface FeatureExplanation {
  feature: string;
  shap_value: number;
  direction: string;
}
