import { useState } from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import LoanForm from "./components/LoanForm";
import PredictionCard from "./components/PredictionCard";
import type { PredictionResponse } from "./types/loan";

export default function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  return <div className="app-shell"><Header /><main className="content"><section className="intro"><span className="eyebrow">Loan application assessment</span><h2>Estimate default risk before making a lending decision.</h2><p>Enter the applicant details and the model will return its predicted default risk.</p></section><div className="workspace"><LoanForm onPrediction={setPrediction} /><PredictionCard prediction={prediction} /></div></main><Footer /></div>;
}
