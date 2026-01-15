import { useState } from 'react';
import DatasetsTab from './components/DatasetsTab';
import DesignTab from './components/DesignTab';
import TrainTab from './components/TrainTab';
import ModelsTab from './components/ModelsTab';
import PredictTab from './components/PredictTab';
import './App.css';

type Tab = 'datasets' | 'design' | 'train' | 'models' | 'predict';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('datasets');

  return (
    <div className="app">
      <header className="header">
        <h1>whitematter</h1>
        <p className="subtitle">Custom ML Training Platform</p>
      </header>

      <nav className="tabs">
        <button
          className={`tab ${activeTab === 'datasets' ? 'active' : ''}`}
          onClick={() => setActiveTab('datasets')}
        >
          Datasets
        </button>
        <button
          className={`tab ${activeTab === 'design' ? 'active' : ''}`}
          onClick={() => setActiveTab('design')}
        >
          Design
        </button>
        <button
          className={`tab ${activeTab === 'train' ? 'active' : ''}`}
          onClick={() => setActiveTab('train')}
        >
          Presets
        </button>
        <button
          className={`tab ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => setActiveTab('models')}
        >
          Models
        </button>
        <button
          className={`tab ${activeTab === 'predict' ? 'active' : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          Predict
        </button>
      </nav>

      <main className="content">
        {activeTab === 'datasets' && <DatasetsTab />}
        {activeTab === 'design' && <DesignTab />}
        {activeTab === 'train' && <TrainTab />}
        {activeTab === 'models' && <ModelsTab />}
        {activeTab === 'predict' && <PredictTab />}
      </main>
    </div>
  );
}

export default App;
