import React, { useState, useEffect } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Layers, CheckCircle2, TrendingUp, Cpu, Award } from 'lucide-react';

export default function Ablation() {
  const [ablationData, setAblationData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');

  useEffect(() => {
    fetch('http://localhost:8000/api/ablation')
      .then((res) => res.json())
      .then((data) => {
        setAblationData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error loading ablation data:', err);
        setLoading(false);
      });
  }, []);

  const chartData = ablationData.map((d) => ({
    name: d.Model.replace(' (ResNet-18)', '').replace('Proposed (Multi-Scale + SE + CORAL)', 'Proposed Full'),
    Accuracy: (d.accuracy * 100).toFixed(1),
    Precision: (d.precision * 100).toFixed(1),
    Sensitivity: (d.sensitivity * 100).toFixed(1),
    Specificity: (d.specificity * 100).toFixed(1),
    F1: (d.f1_score * 100).toFixed(1),
    ROCAUC: (d.roc_auc * 100).toFixed(1),
    BrierScore: (d.brier_score * 100).toFixed(1),
  }));

  const metricsOptions = [
    { key: 'accuracy', label: 'Accuracy', color: 'var(--accent-primary)' },
    { key: 'sensitivity', label: 'Sensitivity / Recall', color: 'var(--danger)' },
    { key: 'specificity', label: 'Specificity', color: 'var(--success)' },
    { key: 'f1_score', label: 'F1-Score', color: 'var(--accent-secondary)' },
    { key: 'roc_auc', label: 'ROC-AUC', color: '#8b5cf6' },
    { key: 'brier_score', label: 'Brier Score (Lower is better)', color: '#f59e0b' },
  ];

  return (
    <div className="ablation-container">
      {/* Header Info */}
      <div className="card mb-8">
        <div className="card-header" style={{ marginBottom: 0 }}>
          <h2 className="card-title">
            <Layers size={20} />
            Component Ablation Study & Architectural Impact
          </h2>
          <span className="badge badge-success">Evaluated on Held-out Patient Validation Set</span>
        </div>
      </div>

      {/* Metric Selector & Main Chart */}
      <div className="mb-8">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Comparative Performance Metric</h2>
            <div className="flex gap-2">
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.4rem 0.8rem',
                  fontSize: '0.875rem'
                }}
              >
                {metricsOptions.map((opt) => (
                  <option key={opt.key} value={opt.key}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <BarChart
                data={ablationData.map((d) => ({
                  name: d.Model.replace(' (ResNet-18)', '').replace('Proposed (Multi-Scale + SE + CORAL)', 'Proposed Full'),
                  Value: parseFloat((d[selectedMetric] * 100).toFixed(1)),
                }))}
                margin={{ top: 20, right: 30, left: 10, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="name" stroke="var(--text-secondary)" tick={{ fontSize: 11 }} angle={-10} textAnchor="end" />
                <YAxis domain={[0, 100]} stroke="var(--text-secondary)" tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: 'var(--radius-md)' }}
                  formatter={(value) => [`${value}%`, selectedMetric.toUpperCase()]}
                />
                <Bar dataKey="Value" fill="var(--accent-primary)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Comprehensive Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Complete 7-Metric Benchmark Table</h2>
          <span className="badge badge-info">Zero Test Label Leakage</span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Model Architecture</th>
                <th style={{ padding: '0.75rem 1rem' }}>Accuracy</th>
                <th style={{ padding: '0.75rem 1rem' }}>Precision</th>
                <th style={{ padding: '0.75rem 1rem' }}>Sensitivity</th>
                <th style={{ padding: '0.75rem 1rem' }}>Specificity</th>
                <th style={{ padding: '0.75rem 1rem' }}>F1-Score</th>
                <th style={{ padding: '0.75rem 1rem' }}>ROC-AUC</th>
                <th style={{ padding: '0.75rem 1rem' }}>Brier Score</th>
              </tr>
            </thead>
            <tbody>
              {ablationData.map((row, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid var(--border-color)',
                    backgroundColor: row.Model.includes('Proposed') ? 'rgba(14, 165, 233, 0.08)' : 'transparent',
                    fontWeight: row.Model.includes('Proposed') ? 600 : 400
                  }}
                >
                  <td style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {row.Model.includes('Proposed') && <CheckCircle2 size={16} color="var(--accent-primary)" />}
                    {row.Model}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.precision * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.sensitivity * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.specificity * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.f1_score * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{(row.roc_auc * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.75rem 1rem', color: row.brier_score < 0.08 ? 'var(--success)' : 'var(--text-secondary)' }}>
                    {row.brier_score.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
