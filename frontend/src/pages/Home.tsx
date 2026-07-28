import { useState } from "react";
import LoanForm from "../components/LoanForm";
import PredictionCard from "../components/PredictionCard";
import type { PredictionResponse } from "../types/loan";

/** Reusable page view for callers that want the predictor without the app shell. */
export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  return <main className="content"><div className="workspace"><LoanForm onPrediction={setPrediction} /><PredictionCard prediction={prediction} /></div></main>;
}
