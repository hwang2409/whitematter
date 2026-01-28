import { useState, useEffect } from 'react';
import * as api from '../api';

export default function ModelsTab() {
  const [models, setModels] = useState<api.Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<api.Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    setLoading(true);
    try {
      const data = await api.getModels();
      setModels(data);
    } catch (e) {
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Are you sure you want to delete this model?')) return;

    try {
      await api.deleteModel(id);
      setModels(models.filter((m) => m.id !== id));
      if (selectedModel?.id === id) {
        setSelectedModel(null);
      }
    } catch (e) {
      setError('Failed to delete model');
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleString();
  }

  function getStatusClass(status: string) {
    switch (status) {
      case 'completed':
        return 'status-completed';
      case 'running':
        return 'status-running';
      case 'failed':
        return 'status-failed';
      case 'cancelled':
        return 'status-cancelled';
      default:
        return 'status-pending';
    }
  }

  async function copyEndpoint(modelId: string) {
    const url = `http://localhost:8080/api/${modelId}/predict`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  }

  if (loading) {
    return <div className="models-tab"><p>Loading models...</p></div>;
  }

  return (
    <div className="models-tab">
      <div className="models-header">
        <h2>Trained Models</h2>
        <button className="btn" onClick={loadModels}>Refresh</button>
      </div>

      {error && <div className="error">{error}</div>}

      {models.length === 0 ? (
        <p className="empty-state">No models yet. Train one in the Train tab!</p>
      ) : (
        <div className="models-layout">
          <div className="models-list">
            {models.map((model) => (
              <div
                key={model.id}
                className={`model-card ${selectedModel?.id === model.id ? 'selected' : ''}`}
                onClick={() => setSelectedModel(model)}
              >
                <div className="model-card-header">
                  <h3>{model.name}</h3>
                  <span className={`status-badge ${getStatusClass(model.status)}`}>
                    {model.status}
                  </span>
                </div>
                <div className="model-card-info">
                  <span>{model.dataset}</span>
                  <span>{model.architecture}</span>
                </div>
                <div className="model-card-stats">
                  <span>{model.epochs_trained} epochs</span>
                  <span>{model.best_accuracy.toFixed(2)}% acc</span>
                </div>
              </div>
            ))}
          </div>

          {selectedModel && (
            <div className="model-details">
              <h3>{selectedModel.name}</h3>

              <div className="details-grid">
                <div className="detail-item">
                  <span className="label">ID</span>
                  <span className="value">{selectedModel.id}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Dataset</span>
                  <span className="value">{selectedModel.dataset}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Architecture</span>
                  <span className="value">{selectedModel.architecture}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Created</span>
                  <span className="value">{formatDate(selectedModel.created_at)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Epochs Trained</span>
                  <span className="value">{selectedModel.epochs_trained}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Best Accuracy</span>
                  <span className="value">{selectedModel.best_accuracy.toFixed(2)}%</span>
                </div>
                <div className="detail-item">
                  <span className="label">Status</span>
                  <span className={`value ${getStatusClass(selectedModel.status)}`}>
                    {selectedModel.status}
                  </span>
                </div>
              </div>

              {selectedModel.training_history.length > 0 && (
                <div className="training-history">
                  <h4>Training History</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Epoch</th>
                        <th>Loss</th>
                        <th>Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedModel.training_history.map((h) => (
                        <tr key={h.epoch}>
                          <td>{h.epoch}</td>
                          <td>{h.loss.toFixed(4)}</td>
                          <td>{h.accuracy.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedModel.status === 'completed' && (
                <div className="api-endpoint">
                  <h4>API Endpoint</h4>
                  <div className="endpoint-row">
                    <code>POST /api/{selectedModel.id}/predict</code>
                    <button
                      className="btn copy-btn"
                      onClick={() => copyEndpoint(selectedModel.id)}
                    >
                      {copied ? 'Copied!' : 'Copy URL'}
                    </button>
                  </div>
                  <details className="curl-example">
                    <summary>cURL Example</summary>
                    <pre>curl -X POST -F "file=@image.jpg" \{'\n'}  http://localhost:8080/api/{selectedModel.id}/predict</pre>
                  </details>
                </div>
              )}

              <div className="model-actions">
                <button
                  className="btn danger"
                  onClick={() => handleDelete(selectedModel.id)}
                >
                  Delete Model
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
