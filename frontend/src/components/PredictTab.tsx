import { useState, useEffect, useRef } from 'react';
import * as api from '../api';

export default function PredictTab() {
  const [models, setModels] = useState<api.Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<api.PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    try {
      const data = await api.getModels();
      const completedModels = data.filter((m) => m.status === 'completed');
      setModels(completedModels);
      if (completedModels.length > 0 && !selectedModelId) {
        setSelectedModelId(completedModels[0].id);
      }
    } catch (e) {
      setError('Failed to load models');
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      setError('');

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(f);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) {
      setFile(f);
      setResult(null);
      setError('');

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(f);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  async function handlePredict() {
    if (!file || !selectedModelId) {
      setError('Please select a model and upload an image');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const prediction = await api.predict(selectedModelId, file);
      setResult(prediction);
    } catch (e) {
      setError('Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  const selectedModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="predict-tab">
      <h2>Make Predictions</h2>

      {error && <div className="error">{error}</div>}

      <div className="form-group">
        <label>Model</label>
        <select
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={loading}
        >
          {models.length === 0 ? (
            <option value="">No trained models available</option>
          ) : (
            models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.dataset} - {m.best_accuracy.toFixed(1)}% acc)
              </option>
            ))
          )}
        </select>
        {selectedModel && (
          <p className="help-text">
            {selectedModel.architecture} trained on {selectedModel.dataset}
          </p>
        )}
      </div>

      <div
        className={`drop-zone ${preview ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !preview && fileInputRef.current?.click()}
      >
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <button className="clear-btn" onClick={clearImage}>
              &times;
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <p>Drop an image here or click to upload</p>
            <p className="hint">Supports PNG, JPG, JPEG</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          hidden
        />
      </div>

      <button
        className="btn primary"
        onClick={handlePredict}
        disabled={!file || !selectedModelId || loading}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          <div className="main-prediction">
            <span className="class-name">{result.class_name}</span>
            <span className="confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>

          <div className="probabilities">
            <h4>All Probabilities</h4>
            {Object.entries(result.probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([className, prob]) => (
                <div key={className} className="prob-bar">
                  <span className="prob-label">{className}</span>
                  <div className="prob-track">
                    <div
                      className="prob-fill"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
