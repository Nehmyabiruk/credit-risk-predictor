import axios from "axios";
import type { LoanData, PredictionResponse } from "../types/loan";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000",
});

export async function predictLoan(
  data: LoanData
): Promise<PredictionResponse> {

  const response = await api.post<PredictionResponse>(
    "/predict",
    data
  );

  return response.data;
}
