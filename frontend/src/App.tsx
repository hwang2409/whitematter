import { useState, useEffect } from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import DatasetsTab from './components/DatasetsTab';
import TrainTab from './components/TrainTab';
import DesignHelper from './components/DesignHelper';
import ModelsTab from './components/ModelsTab';
import PredictTab from './components/PredictTab';
import { useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import { getCustomDatasets, getModels } from './api';
import type { CustomDataset, Model, Architecture } from './api';
import './App.css';

type Step = 'data' | 'train' | 'models' | 'predict';

interface DesignHelperContext {
  datasetType?: string;
  architecture?: Architecture | null;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

function App() {
  const { user, loading, logout } = useAuth();
  const [activeStep, setActiveStep] = useState<Step>('data');
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [designHelperOpen, setDesignHelperOpen] = useState(false);
  const [designHelperContext, setDesignHelperContext] = useState<DesignHelperContext>({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  const isRegister = typeof window !== 'undefined' && window.location.hash === '#register';

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [datasetsData, modelsData] = await Promise.all([
        getCustomDatasets(),
        getModels()
      ]);
      setDatasets(datasetsData);
      setModels(modelsData);
    } catch (err) {
      console.error('Failed to load data:', err);
    }
  };

  const readyDatasets = datasets.filter(d => d.status === 'ready');
  const completedModels = models.filter(m => m.status === 'completed');

  const getStepStatus = (step: Step): 'complete' | 'active' | 'pending' => {
    if (step === activeStep) return 'active';
    switch (step) {
      case 'data':
        return readyDatasets.length > 0 ? 'complete' : 'pending';
      case 'train':
        return completedModels.length > 0 ? 'complete' : 'pending';
      case 'models':
        return completedModels.length > 0 ? 'complete' : 'pending';
      case 'predict':
        return 'pending';
      default:
        return 'pending';
    }
  };

  // Close sidebar when switching away from train tab
  useEffect(() => {
    if (activeStep !== 'train') {
      setDesignHelperOpen(false);
    }
  }, [activeStep]);

  if (loading) {
    return (
      <div className="app" style={{ alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <p>Loading…</p>
      </div>
    );
  }
  if (!user) {
    return (
      <ErrorBoundary>
        {isRegister ? <RegisterPage /> : <LoginPage />}
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary>
    <div className={`app ${designHelperOpen ? 'with-sidebar' : ''}`}>
      <header className="header">
        <h1>whitematter</h1>
        <div className="header-actions">
          <span className="header-email">{user.email}</span>
          <button type="button" className="header-logout" onClick={logout}>Log out</button>
        </div>
      </header>

      {/* Workflow Navigation */}
      <nav className="workflow-nav">
        <button
          className={`workflow-step ${getStepStatus('data')}`}
          onClick={() => setActiveStep('data')}
        >
          <span className="step-num">{getStepStatus('data') === 'complete' ? '✓' : '1'}</span>
          <span className="step-label">Data</span>
          {readyDatasets.length > 0 && (
            <span className="step-badge">{readyDatasets.length}</span>
          )}
        </button>

        <div className="workflow-connector" />

        <button
          className={`workflow-step ${getStepStatus('train')}`}
          onClick={() => setActiveStep('train')}
        >
          <span className="step-num">{getStepStatus('train') === 'complete' ? '✓' : '2'}</span>
          <span className="step-label">Train</span>
        </button>

        <div className="workflow-connector" />

        <button
          className={`workflow-step ${getStepStatus('models')}`}
          onClick={() => setActiveStep('models')}
        >
          <span className="step-num">{completedModels.length > 0 ? '✓' : '3'}</span>
          <span className="step-label">Models</span>
          {completedModels.length > 0 && (
            <span className="step-badge">{completedModels.length}</span>
          )}
        </button>

        <div className="workflow-connector" />

        <button
          className={`workflow-step ${getStepStatus('predict')}`}
          onClick={() => setActiveStep('predict')}
        >
          <span className="step-num">4</span>
          <span className="step-label">Predict</span>
        </button>
      </nav>

      <div className="app-body">
        {/* Main Content */}
        <main className="content">
          {activeStep === 'data' && (
            <DatasetsTab
              onDatasetSelect={(id) => setSelectedDataset(id)}
              onUpdate={loadData}
            />
          )}
          {activeStep === 'train' && (
            <TrainTab
              datasets={readyDatasets}
              selectedDataset={selectedDataset}
              onDatasetChange={setSelectedDataset}
              onTrainingComplete={loadData}
              helperOpen={designHelperOpen}
              onHelperToggle={setDesignHelperOpen}
              onHelperContextChange={setDesignHelperContext}
            />
          )}
          {activeStep === 'models' && (
            <ModelsTab
              onModelSelect={(id) => setSelectedModel(id)}
              onUpdate={loadData}
            />
          )}
          {activeStep === 'predict' && (
            <PredictTab
              models={completedModels}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          )}
        </main>

        {/* Design Helper Sidebar */}
        {designHelperOpen && (
          <aside className="app-sidebar">
            <div className="sidebar-header">
              <h3>AI Design Assistant</h3>
              <button className="sidebar-close" onClick={() => setDesignHelperOpen(false)}>
                &times;
              </button>
            </div>
            <div className="sidebar-body">
              <DesignHelper
                datasetType={designHelperContext.datasetType}
                currentArchitecture={designHelperContext.architecture}
                messages={chatMessages}
                onMessagesChange={setChatMessages}
              />
            </div>
          </aside>
        )}
      </div>
    </div>
    </ErrorBoundary>
  );
}

export default App;
