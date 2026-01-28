import { useState, useEffect } from 'react';
import * as api from '../api';
import TrainingChart from './TrainingChart';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export default function DesignTab() {
  const [datasets, setDatasets] = useState<api.CustomDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [architecture, setArchitecture] = useState<api.Architecture | null>(null);
  const [explanation, setExplanation] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [refining, setRefining] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainingJob, setTrainingJob] = useState<api.CustomTrainJob | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    let interval: number;
    if (trainingJob && ['pending', 'compiling', 'training'].includes(trainingJob.status)) {
      interval = setInterval(async () => {
        try {
          const status = await api.getCustomTrainingStatus(trainingJob.job_id);
          setTrainingJob(status);

          // Accumulate training history
          if (status.epoch !== undefined && status.loss !== undefined && status.accuracy !== undefined) {
            setTrainingHistory((prev) => {
              const existing = prev.find((h) => h.epoch === status.epoch);
              if (existing) return prev;
              return [...prev, { epoch: status.epoch!, loss: status.loss!, accuracy: status.accuracy! }];
            });
          }

          if (!['pending', 'compiling', 'training'].includes(status.status)) {
            setTraining(false);
          }
        } catch (e) {
          console.error('Failed to get training status:', e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [trainingJob?.job_id, trainingJob?.status]);

  async function loadDatasets() {
    try {
      const data = await api.getCustomDatasets();
      const readyDatasets = data.filter((d) => d.status === 'ready');
      setDatasets(readyDatasets);
      if (readyDatasets.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(readyDatasets[0].id);
      }
    } catch (e) {
      setError('Failed to load datasets');
    }
  }

  async function handleSuggest() {
    if (!selectedDatasetId || !prompt.trim()) {
      setError('Please select a dataset and describe what you want');
      return;
    }

    setLoading(true);
    setError('');
    setArchitecture(null);
    setValidation(null);

    try {
      const result = await api.suggestArchitecture(selectedDatasetId, prompt.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);

      // Auto-validate
      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to get suggestion');
    } finally {
      setLoading(false);
    }
  }

  async function handleRefine() {
    if (!architecture || !feedback.trim()) {
      setError('Please provide feedback for refinement');
      return;
    }

    setRefining(true);
    setError('');

    try {
      const result = await api.refineArchitecture(architecture, feedback.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);
      setFeedback('');

      // Auto-validate
      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to refine');
    } finally {
      setRefining(false);
    }
  }

  async function handleTrain() {
    if (!selectedDatasetId || !architecture) {
      setError('Missing dataset or architecture');
      return;
    }

    if (validation && !validation.valid) {
      setError('Please fix validation errors before training');
      return;
    }

    setTraining(true);
    setError('');
    setTrainingHistory([]);

    try {
      const job = await api.startCustomTraining(selectedDatasetId, architecture);
      setTrainingJob(job);
    } catch (e: any) {
      setError(e.message || 'Failed to start training');
      setTraining(false);
    }
  }

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="design-tab">
      <h2>Design Your Model</h2>
      <p className="subtitle">Describe what you want in natural language and get an architecture suggestion.</p>

      {error && <div className="error">{error}</div>}

      {/* Dataset Selection */}
      <div className="form-group">
        <label>Dataset</label>
        <select
          value={selectedDatasetId}
          onChange={(e) => setSelectedDatasetId(e.target.value)}
          disabled={loading || training}
        >
          {datasets.length === 0 ? (
            <option value="">No datasets available - upload one first</option>
          ) : (
            datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.data_type}, {d.num_classes} classes)
              </option>
            ))
          )}
        </select>
        {selectedDataset && (
          <p className="help-text">
            Input: [{selectedDataset.input_shape.join('x')}], {selectedDataset.total_samples} samples
          </p>
        )}
      </div>

      {/* Prompt Input */}
      <div className="form-group">
        <label>Describe Your Model</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., I want a CNN with good accuracy for image classification. Use batch normalization and dropout for regularization."
          rows={4}
          disabled={loading || training}
        />
      </div>

      <button
        className="btn primary"
        onClick={handleSuggest}
        disabled={!selectedDatasetId || !prompt.trim() || loading}
      >
        {loading ? 'Generating...' : 'Generate Architecture'}
      </button>

      {/* Architecture Display */}
      {architecture && (
        <div className="architecture-section">
          <h3>Suggested Architecture: {architecture.name}</h3>

          {explanation && (
            <div className="explanation">
              <p>{explanation}</p>
            </div>
          )}

          {/* Validation */}
          {validation && (
            <div className={`validation ${validation.valid ? 'valid' : 'invalid'}`}>
              {validation.valid ? (
                <span className="validation-status">Architecture is valid</span>
              ) : (
                <>
                  <span className="validation-status">Validation failed</span>
                  {validation.errors.map((err, i) => (
                    <p key={i} className="validation-error">{err}</p>
                  ))}
                </>
              )}
              {validation.warnings.map((warn, i) => (
                <p key={i} className="validation-warning">{warn}</p>
              ))}
            </div>
          )}

          {/* Layer Summary */}
          <div className="layers-section">
            <h4>Layers ({architecture.layers.length})</h4>
            <div className="layers-list">
              {architecture.layers.map((layer, i) => (
                <div key={i} className="layer-item">
                  <span className="layer-index">{i + 1}</span>
                  <span className="layer-type">{layer.type}</span>
                  <span className="layer-params">
                    {Object.entries(layer.params)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(', ')}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Training Config */}
          <div className="training-config">
            <h4>Training Configuration</h4>
            <div className="config-grid">
              <div className="config-item">
                <span className="label">Optimizer</span>
                <span className="value">
                  {architecture.training.optimizer.type} (lr={architecture.training.optimizer.params.learning_rate})
                </span>
              </div>
              <div className="config-item">
                <span className="label">Scheduler</span>
                <span className="value">{architecture.training.scheduler.type}</span>
              </div>
              <div className="config-item">
                <span className="label">Epochs</span>
                <span className="value">{architecture.training.epochs}</span>
              </div>
              <div className="config-item">
                <span className="label">Batch Size</span>
                <span className="value">{architecture.training.batch_size}</span>
              </div>
            </div>
          </div>

          {/* JSON View */}
          <details className="json-details">
            <summary>View Full JSON</summary>
            <pre className="json-view">{JSON.stringify(architecture, null, 2)}</pre>
          </details>

          {/* Refinement */}
          <div className="refinement-section">
            <h4>Refine Architecture</h4>
            <div className="form-group">
              <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="e.g., Add more dropout layers, use a smaller learning rate, increase the number of filters..."
                rows={2}
                disabled={refining || training}
              />
            </div>
            <button
              className="btn"
              onClick={handleRefine}
              disabled={!feedback.trim() || refining}
            >
              {refining ? 'Refining...' : 'Refine'}
            </button>
          </div>

          {/* Train Button */}
          <div className="train-section">
            <button
              className="btn primary large"
              onClick={handleTrain}
              disabled={training || (validation !== null && !validation.valid)}
            >
              {training ? 'Training...' : 'Start Training'}
            </button>
          </div>
        </div>
      )}

      {/* Training Status */}
      {trainingJob && (
        <div className="training-status">
          <h3>Training Progress</h3>
          <div className="status-grid">
            <div className="status-item">
              <span className="label">Status</span>
              <span className={`value status-${trainingJob.status}`}>{trainingJob.status}</span>
            </div>
            {trainingJob.epoch !== undefined && (
              <div className="status-item">
                <span className="label">Epoch</span>
                <span className="value">{trainingJob.epoch} / {trainingJob.total_epochs}</span>
              </div>
            )}
            {trainingJob.loss !== undefined && (
              <div className="status-item">
                <span className="label">Loss</span>
                <span className="value">{trainingJob.loss.toFixed(4)}</span>
              </div>
            )}
            {trainingJob.accuracy !== undefined && (
              <div className="status-item">
                <span className="label">Accuracy</span>
                <span className="value">{trainingJob.accuracy.toFixed(2)}%</span>
              </div>
            )}
          </div>
          <p className="message">{trainingJob.message}</p>
          {trainingJob.status === 'training' && trainingJob.epoch !== undefined && (
            <div className="progress-bar">
              <div
                className="progress"
                style={{ width: `${(trainingJob.epoch / (trainingJob.total_epochs || 1)) * 100}%` }}
              />
            </div>
          )}
          {trainingHistory.length > 0 && (
            <TrainingChart data={trainingHistory} />
          )}
          {trainingJob.status === 'completed' && (
            <div className="success-message">
              <p>Training completed! Model ID: <code>{trainingJob.model_id}</code></p>
              <p className="api-hint">
                API endpoint: <code>POST /api/{trainingJob.model_id}/predict</code>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
