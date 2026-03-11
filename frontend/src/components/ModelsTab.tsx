"use client";
import { useState, useEffect } from 'react';
import * as api from '@/api';
import ConfirmDialog from './ConfirmDialog';
import Toast, { useToast } from './Toast';

interface Props {
  onModelSelect?: (id: string | null) => void;
  onUpdate?: () => void;
}

export default function ModelsTab({ onModelSelect, onUpdate }: Props) {
  const [models, setModels] = useState<api.Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<api.Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  // Text generation state
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);

  // Confirm dialog state
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Toast notifications
  const toast = useToast();

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    setLoading(true);
    try {
      const data = await api.getModels();
      setModels(data);
      onUpdate?.();
    } catch (e) {
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  }

  function handleSelectModel(model: api.Model) {
    setSelectedModel(model);
    onModelSelect?.(model.id);
  }

  async function confirmDelete(id: string) {
    try {
      await api.deleteModel(id);
      setModels(models.filter((m) => m.id !== id));
      if (selectedModel?.id === id) {
        setSelectedModel(null);
      }
    } catch (e) {
      setError('Failed to delete model');
    } finally {
      setDeleteConfirm(null);
    }
  }

  async function handleResume(id: string) {
    try {
      setError('');
      const result = await api.resumeTraining(id);
      // Update the model in the list to show it's running
      setModels(models.map((m) =>
        m.id === id ? { ...m, status: 'running' as const } : m
      ));
      if (selectedModel?.id === id) {
        setSelectedModel({ ...selectedModel, status: 'running' });
      }
      toast.success(`Resuming training from epoch ${result.start_epoch}. Check the Train tab for progress.`);
    } catch (e: any) {
      toast.error(e.message || 'Failed to resume training');
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleString();
  }

  function formatModelName(name: string) {
    // Clean up model names like "custom_minigpt_bf230670" to "MiniGPT"
    const parts = name.replace('custom_', '').split('_');
    if (parts.length > 1) {
      // Remove the hash suffix and capitalize
      const baseName = parts.slice(0, -1).join('_');
      return baseName.split(/[-_]/).map(w =>
        w.charAt(0).toUpperCase() + w.slice(1)
      ).join(' ');
    }
    return name;
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

  function isTextModel(model: api.Model) {
    // Check if this is a text/language model by name or architecture
    const name = model.name.toLowerCase();
    const arch = model.architecture.toLowerCase();
    return model.dataset.startsWith('custom:') &&
      (name.includes('gpt') || name.includes('language') || name.includes('text') ||
       arch.includes('gpt') || arch.includes('language') || arch.includes('text') || arch.includes('lstm'));
  }

  async function handleGenerate() {
    if (!selectedModel || !prompt.trim()) return;

    setGenerating(true);
    setError('');
    setGeneratedText('');

    try {
      const result = await api.generateText(selectedModel.id, {
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        temperature,
      });
      // Clean up the output (remove debug info)
      let text = result.generated_text;
      if (text.includes('\n') && text.startsWith('Model loaded:')) {
        text = text.split('\n').slice(1).join('\n');
      }
      setGeneratedText(text);
    } catch (e: any) {
      setError(e.message || 'Failed to generate text');
    } finally {
      setGenerating(false);
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
                onClick={() => handleSelectModel(model)}
              >
                <div className="model-card-header">
                  <h3>{formatModelName(model.name)}</h3>
                  <span className={`status-badge ${getStatusClass(model.status)}`}>
                    {model.status}
                  </span>
                </div>
                <div className="model-card-info">
                  <span>{model.dataset.startsWith('custom:') ? 'Custom Dataset' : model.dataset}</span>
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
              <h3>{formatModelName(selectedModel.name)}</h3>

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

              {selectedModel.status === 'completed' && isTextModel(selectedModel) && (
                <div className="text-generation-section">
                  <h4>Generate Text</h4>
                  <div className="form-group">
                    <label>Prompt</label>
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Enter a starting prompt... (e.g., 'HAMLET:\nTo be or not to be')"
                      rows={3}
                      disabled={generating}
                    />
                  </div>
                  <div className="form-row">
                    <div className="form-group half">
                      <label>Temperature ({temperature})</label>
                      <input
                        type="range"
                        min="0.1"
                        max="1.5"
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        disabled={generating}
                      />
                    </div>
                    <div className="form-group half">
                      <label>Max Tokens</label>
                      <input
                        type="number"
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                        min={10}
                        max={500}
                        disabled={generating}
                      />
                    </div>
                  </div>
                  <button
                    className="btn primary"
                    onClick={handleGenerate}
                    disabled={generating || !prompt.trim()}
                  >
                    {generating ? 'Generating...' : 'Generate'}
                  </button>

                  {generatedText && (
                    <div className="generated-output">
                      <h4>Generated Text</h4>
                      <pre>{generatedText}</pre>
                    </div>
                  )}
                </div>
              )}

              {selectedModel.status === 'completed' && !isTextModel(selectedModel) && (
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
                {(selectedModel.status === 'failed' || selectedModel.status === 'cancelled') && (
                  <button
                    className="btn primary"
                    onClick={() => handleResume(selectedModel.id)}
                  >
                    Resume Training
                  </button>
                )}
                <button
                  className="btn danger"
                  onClick={() => setDeleteConfirm(selectedModel.id)}
                >
                  Delete Model
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <ConfirmDialog
        isOpen={deleteConfirm !== null}
        title="Delete Model"
        message="Are you sure you want to delete this model? This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={() => deleteConfirm && confirmDelete(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
      />

      <Toast toasts={toast.toasts} onDismiss={toast.dismissToast} />
    </div>
  );
}
