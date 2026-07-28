import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import type { LoanData, PredictionResponse } from "../types/loan";
import { predictLoan } from "../services/api";

interface Props { onPrediction: (result: PredictionResponse) => void; }
const initialForm: LoanData = { person_age: 25, person_gender: "male", person_education: "Bachelor", person_income: 50000, person_emp_exp: 3, person_home_ownership: "RENT", loan_amnt: 10000, loan_intent: "PERSONAL", loan_int_rate: 12, loan_percent_income: 0.2, cb_person_cred_hist_length: 4, credit_score: 650, previous_loan_defaults_on_file: "No" };
const numberFields: Array<[keyof LoanData, string, string]> = [["person_age", "Age", "1"], ["person_income", "Annual income", "1000"], ["person_emp_exp", "Employment experience (years)", "1"], ["loan_amnt", "Loan amount", "100"], ["loan_int_rate", "Interest rate (%)", "0.1"], ["loan_percent_income", "Loan / income ratio", "0.01"], ["credit_score", "Credit score", "1"], ["cb_person_cred_hist_length", "Credit history (years)", "1"]];
const selectFields: Array<[keyof LoanData, string, string[]]> = [["person_gender", "Gender", ["male", "female"]], ["person_education", "Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"]], ["person_home_ownership", "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]], ["loan_intent", "Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]], ["previous_loan_defaults_on_file", "Previous defaults", ["No", "Yes"]]];

export default function LoanForm({ onPrediction }: Props) {
  const [form, setForm] = useState<LoanData>(initialForm); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null);
  const handleChange = (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => { const { name, value, type } = event.target; setForm(current => ({ ...current, [name]: type === "number" ? Number(value) : value } as LoanData)); };
  async function handleSubmit(event: FormEvent) { event.preventDefault(); setLoading(true); setError(null); try { onPrediction(await predictLoan(form)); } catch (err) { console.error(err); setError("We could not get a prediction. Make sure the API is running on port 8000 and try again."); } finally { setLoading(false); } }
  return <form onSubmit={handleSubmit} className="loan-form card"><div><span className="form-kicker">Application details</span><h3>Tell us about the applicant</h3></div><div className="form-grid">{numberFields.map(([name, label, step]) => <label key={name}><span>{label}</span><input className="field" type="number" name={name} value={form[name] as number} step={step} required onChange={handleChange} /></label>)}{selectFields.map(([name, label, options]) => <label key={name}><span>{label}</span><select className="field" name={name} value={form[name] as string} onChange={handleChange}>{options.map(option => <option key={option}>{option}</option>)}</select></label>)}</div>{error && <p className="form-error" role="alert">{error}</p>}<button className="submit-button" disabled={loading}>{loading ? "Predicting…" : "Predict credit risk"}</button></form>;
}
