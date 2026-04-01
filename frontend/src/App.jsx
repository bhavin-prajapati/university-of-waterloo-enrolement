import React, { useState, useCallback, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { FIELD_OPTIONS, FIELD_LABELS, DEFAULT_PARAMS } from "./options";
import MultiSelect from "./MultiSelect";

const API_BASE = "/api";
const MAX_COMBOS = 500;
const MODELS = ["Linear Regression", "Random Forest", "Gradient-Boosted Trees"];
const COLORS = ["#8884d8", "#82ca9d", "#ff7300", "#0088fe", "#ff4444"];

/* ── helpers ─────────────────────────────────────────────────────────────── */

/** Cartesian product of all value-arrays in `obj`. Returns flat list of objects. */
function cartesian(obj) {
  const keys = Object.keys(obj);
  const arrays = keys.map((k) => obj[k]);
  return arrays
    .reduce((acc, vals) => acc.flatMap((combo) => vals.map((v) => [...combo, v])), [[]])
    .map((combo) => {
      const out = {};
      keys.forEach((k, i) => (out[k] = combo[i]));
      return out;
    });
}

/** Group predictions by `key`, compute average headcount per group. */
function summarize(predictions, key) {
  const groups = {};
  for (const p of predictions) {
    const k = p[key];
    if (!groups[k]) groups[k] = [];
    groups[k].push(p.predicted_student_headcount);
  }
  return Object.entries(groups)
    .map(([label, vals]) => ({
      name: label,
      headcount: Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 100) / 100,
    }))
    .sort((a, b) => (a.name > b.name ? 1 : -1));
}

async function batchPredict(requests, modelName) {
  const url = modelName
    ? `${API_BASE}/predict/batch?model_name=${encodeURIComponent(modelName)}`
    : `${API_BASE}/predict/batch`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requests }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

/* ── App ─────────────────────────────────────────────────────────────────── */

export default function App() {
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [modelComparison, setModelComparison] = useState(null);
  const [yearTrend, setYearTrend] = useState(null);
  const [termComparison, setTermComparison] = useState(null);
  const [facultyComparison, setFacultyComparison] = useState(null);

  const comboCount = useMemo(
    () => Object.values(params).reduce((n, arr) => n * arr.length, 1),
    [params]
  );

  const onChange = (field, values) => {
    setParams((p) => ({ ...p, [field]: values }));
  };

  /* ── Run all chart queries ───────────────────────────────────────────── */
  const runPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const combos = cartesian(params);
      if (combos.length > MAX_COMBOS) {
        throw new Error(
          `Too many combinations (${combos.length.toLocaleString()}). ` +
          `Please reduce your selections to at most ${MAX_COMBOS}.`
        );
      }

      // 1. Best-model predictions → dimension charts
      const bestResult = await batchPredict(combos);
      const preds = bestResult.predictions;

      setYearTrend(summarize(preds, "fiscal_year"));
      setTermComparison(summarize(preds, "term_type"));
      setFacultyComparison(summarize(preds, "faculty_group"));

      // 2. Model comparison → one batch per model
      const modelResults = await Promise.all(
        MODELS.map((m) => batchPredict(combos, m))
      );
      setModelComparison(
        modelResults.map((r, i) => {
          const vals = r.predictions.map((p) => p.predicted_student_headcount);
          const avg = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
          return { model: MODELS[i], headcount: Math.round(avg * 100) / 100 };
        })
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [params]);

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <div className="app">
      <header>
        <h1>University of Waterloo — Enrolment Prediction Dashboard</h1>
      </header>

      <div className="layout">
        {/* ── Sidebar form ──────────────────────────────────────────── */}
        <aside className="sidebar">
          <h2>Parameters</h2>
          <p className="combo-count">
            {comboCount.toLocaleString()} combination{comboCount !== 1 ? "s" : ""}
            {comboCount > MAX_COMBOS && (
              <span className="combo-warn"> (max {MAX_COMBOS})</span>
            )}
          </p>

          {Object.keys(FIELD_LABELS).map((field) => (
            <MultiSelect
              key={field}
              label={FIELD_LABELS[field]}
              options={FIELD_OPTIONS[field]}
              selected={params[field]}
              onChange={(vals) => onChange(field, vals)}
            />
          ))}

          <button
            className="predict-btn"
            onClick={runPredictions}
            disabled={loading || comboCount > MAX_COMBOS}
          >
            {loading ? "Running…" : "Predict"}
          </button>
          {error && <p className="error">{error}</p>}
        </aside>

        {/* ── Charts ────────────────────────────────────────────────── */}
        <main className="charts">
          {!modelComparison && !loading && (
            <p className="placeholder">
              Configure parameters and click <strong>Predict</strong> to see
              charts. Use the dropdowns to select multiple values.
            </p>
          )}

          {modelComparison && (
            <section className="chart-card">
              <h3>Model Comparison (avg headcount)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelComparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="headcount" fill={COLORS[0]} name="Avg Headcount" />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          {yearTrend && (
            <section className="chart-card">
              <h3>Headcount Trend by Fiscal Year (avg)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={yearTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="headcount"
                    stroke={COLORS[1]}
                    strokeWidth={2}
                    name="Avg Headcount"
                  />
                </LineChart>
              </ResponsiveContainer>
            </section>
          )}

          {termComparison && (
            <section className="chart-card">
              <h3>Headcount by Term Type (avg)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={termComparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="headcount" fill={COLORS[2]} name="Avg Headcount" />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          {facultyComparison && (
            <section className="chart-card">
              <h3>Headcount by Faculty Group (avg)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={facultyComparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="headcount" fill={COLORS[3]} name="Avg Headcount" />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
