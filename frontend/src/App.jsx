import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Upload, Database, Activity, BarChart2, AlertTriangle, ActivitySquare } from 'lucide-react';
import './index.css';

// Import Views
import Dashboard from './views/Dashboard';
import Predict from './views/Predict';
import DataExplorer from './views/DataExplorer';
import Robustness from './views/Robustness';
import Ablation from './views/Ablation';

function Sidebar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/predict', label: 'PCOS Detection & Grad-CAM', icon: <Upload size={20} /> },
    { path: '/robustness', label: 'Domain Robustness (CORAL)', icon: <Activity size={20} /> },
    { path: '/ablation', label: 'Ablation Study', icon: <BarChart2 size={20} /> },
    { path: '/data', label: 'Dataset Explorer', icon: <Database size={20} /> },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <ActivitySquare className="sidebar-logo-icon" size={28} />
        <span>Auto-PCOS AI</span>
      </div>
      
      <div className="nav-links">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
          >
            {item.icon}
            <span>{item.label}</span>
          </Link>
        ))}
      </div>

      <div style={{ marginTop: 'auto', padding: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        <div>Framework: PyTorch + CORAL</div>
        <div style={{ marginTop: '0.25rem' }}>MC Dropout: 15 Passes</div>
      </div>
    </div>
  );
}


function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <div className="main-content">
          <div className="content-area">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/robustness" element={<Robustness />} />
              <Route path="/ablation" element={<Ablation />} />
              <Route path="/data" element={<DataExplorer />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;
