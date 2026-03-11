import { useState, useEffect } from 'react';
import * as api from '../api';
import TrainingChart from './TrainingChart';

type TrainMode = 'quick' | 'custom';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface Props {
  datasets?: api.CustomDataset[];
  selectedDataset?: string | null;
  onDatasetChange?: (id: string) => void;
  onTrainingComplete?: () => void;
  helperOpen?: boolean;
  onHelperToggle?: (open: boolean) => void;
  onHelperContextChange?: (context: { datasetType?: string; architecture?: api.Architecture | null }) => void;
}

export default function TrainTab({
  datasets: propDatasets,
  selectedDataset,
  onDatasetChange,
  onTrainingComplete,
  helperOpen,
  onHelperToggle,
  onHelperContextChange,
}: Props) {
  // Mode toggle
  const [mode, setMode] = useState<TrainMode>('quick');

  // Dataset state
  const [localDatasets, setLocalDatasets] = useState<api.CustomDataset[]>([]);
  const datasets = propDatasets?.length ? propDatasets : localDatasets;
  const [selectedDatasetId, setSelectedDatasetId] = useState(selectedDataset || '');

  // Quick Start state (presets)
  const [builtInDatasets, setBuiltInDatasets] = useState<api.Dataset[]>([]);
  const [presets, setPresets] = useState<api.Preset[]>([]);
  const [optimizers, setOptimizers] = useState<api.Optimizer[]>([]);
  const [schedulers, setSchedulers] = useState<api.Scheduler[]>([]);
  const [augmentations, setAugmentations] = useState<api.Augmentation[]>([]);
  const [selectedBuiltInDataset, setSelectedBuiltInDataset] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(64);
  const [modelName, setModelName] = useState('');
  const [selectedOptimizer, setSelectedOptimizer] = useState('sgd');
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.0);
  const [selectedScheduler, setSelectedScheduler] = useState('none');
  const [schedulerStepSize, setSchedulerStepSize] = useState(10);
  const [schedulerGamma, setSchedulerGamma] = useState(0.1);
  const [enabledAugs, setEnabledAugs] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Custom Design state
  const [prompt, setPrompt] = useState('');
  const [architecture, setArchitecture] = useState<api.Architecture | null>(null);
  const [explanation, setExplanation] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [feedback, setFeedback] = useState('');
  const [generating, setGenerating] = useState(false);
  const [refining, setRefining] = useState(false);

  // Shared training state
  const [training, setTraining] = useState(false);
  const [trainingJob, setTrainingJob] = useState<api.CustomTrainJob | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);
  const [error, setError] = useState('');

  const currentDataset = datasets.find((d) => d.id === selectedDatasetId);

  // Load data on mount
  useEffect(() => {
    loadDatasets();
    loadQuickStartOptions();
  }, []);

  // Shared handler for status updates from either WebSocket or polling
  function handleStatusUpdate(status: api.CustomTrainJob) {
    setTrainingJob(status);

    const epoch = status.epoch ?? 0;
    const loss = status.loss ?? 0;
    const accuracy = status.accuracy ?? 0;

    if (epoch > 0) {
      setTrainingHistory((prev) => {
        const existing = prev.find((h) => h.epoch === epoch);
        if (existing) return prev;
        return [...prev, { epoch, loss, accuracy }];
      });
    }

    if (!['pending', 'compiling', 'training', 'running'].includes(status.status)) {
      setTraining(false);
      if (status.status === 'completed') {
        onTrainingComplete?.();
      }
    }
  }

  // WebSocket-first with HTTP polling fallback
  useEffect(() => {
    if (!trainingJob || !['pending', 'compiling', 'training', 'running'].includes(trainingJob.status)) {
      return;
    }

    const jobId = trainingJob.job_id;
    let pollInterval: number | undefined;
    let wsHandle: { close: () => void } | null = null;

    function startPolling() {
      pollInterval = window.setInterval(async () => {
        try {
          const status = await api.getCustomTrainingStatus(jobId);
          handleStatusUpdate(status);
        } catch (e) {
          console.error('Failed to get training status:', e);
        }
      }, 2000);
    }

    // Try WebSocket first
    wsHandle = api.createTrainingWebSocket(
      jobId,
      (status) => handleStatusUpdate(status),
      () => {
        // On WebSocket error, fall back to polling
        if (!pollInterval) startPolling();
      },
      () => {
        // On WebSocket close (after max reconnects), fall back to polling
        wsHandle = null;
        if (!pollInterval) startPolling();
      },
    );

    return () => {
      wsHandle?.close();
      if (pollInterval) clearInterval(pollInterval);
    };
    // Only re-subscribe when the job_id changes, not on every status update.
    // The early return handles terminal states.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingJob?.job_id]);

  // Sync external dataset selection
  useEffect(() => {
    if (selectedDataset && selectedDataset !== selectedDatasetId) {
      setSelectedDatasetId(selectedDataset);
    }
  }, [selectedDataset]);

  // Auto-select first custom dataset
  useEffect(() => {
    if (mode === 'custom' && datasets.length > 0 && !selectedDatasetId) {
      const firstId = datasets[0].id;
      setSelectedDatasetId(firstId);
      onDatasetChange?.(firstId);
    }
  }, [datasets, mode]);

  // Update helper context
  useEffect(() => {
    if (mode === 'custom') {
      onHelperContextChange?.({
        datasetType: currentDataset?.data_type,
        architecture: architecture,
      });
    }
  }, [architecture, currentDataset?.data_type, mode]);

  // Filter presets for selected built-in dataset
  const filteredPresets = presets.filter((p) => p.dataset === selectedBuiltInDataset);

  // Update preset when built-in dataset changes
  useEffect(() => {
    const filtered = presets.filter((p) => p.dataset === selectedBuiltInDataset);
    if (filtered.length > 0 && !filtered.find((p) => p.id === selectedPreset)) {
      setSelectedPreset(filtered[0].id);
    }
  }, [selectedBuiltInDataset, presets]);

  async function loadDatasets() {
    if (propDatasets?.length) return;
    try {
      const data = await api.getCustomDatasets();
      const readyDatasets = data.filter((d) => d.status === 'ready');
      setLocalDatasets(readyDatasets);
    } catch (e) {
      console.error('Failed to load datasets:', e);
    }
  }

  async function loadQuickStartOptions() {
    try {
      const [ds, ps, opts, scheds, augs] = await Promise.all([
        api.getDatasets(),
        api.getPresets(),
        api.getOptimizers(),
        api.getSchedulers(),
        api.getAugmentations(),
      ]);
      setBuiltInDatasets(ds);
      setPresets(ps);
      setOptimizers(opts);
      setSchedulers(scheds);
      setAugmentations(augs);

      if (ds.length > 0) setSelectedBuiltInDataset(ds[0].id);
      const filtered = ps.filter((p) => p.dataset === ds[0]?.id);
      if (filtered.length > 0) setSelectedPreset(filtered[0].id);
    } catch (e) {
      console.error('Failed to load options:', e);
    }
  }

  function handleDatasetChange(id: string) {
    setSelectedDatasetId(id);
    onDatasetChange?.(id);
  }

  function toggleAugmentation(id: string) {
    const newSet = new Set(enabledAugs);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setEnabledAugs(newSet);
  }

  // Custom Design handlers
  async function handleGenerate() {
    if (!selectedDatasetId || !prompt.trim()) {
      setError('Please select a dataset and describe what you want');
      return;
    }

    setGenerating(true);
    setError('');
    setArchitecture(null);
    setValidation(null);

    try {
      const result = await api.suggestArchitecture(selectedDatasetId, prompt.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);

      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to get suggestion');
    } finally {
      setGenerating(false);
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

      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to refine');
    } finally {
      setRefining(false);
    }
  }

  // Training handlers
  async function handleTrainCustom() {
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

  async function handleTrainQuickStart() {
    if (!selectedBuiltInDataset || !selectedPreset) {
      setError('Please select dataset and architecture');
      return;
    }

    setError('');
    setTraining(true);
    setTrainingHistory([]);

    const optimizerParams: Record<string, number> = { learning_rate: learningRate };
    if (selectedOptimizer === 'sgd') {
      optimizerParams.momentum = momentum;
    }
    optimizerParams.weight_decay = weightDecay;

    const schedulerParams: Record<string, number> = {};
    if (selectedScheduler === 'step') {
      schedulerParams.step_size = schedulerStepSize;
      schedulerParams.gamma = schedulerGamma;
    } else if (selectedScheduler === 'exponential') {
      schedulerParams.gamma = schedulerGamma;
    }

    const augConfigs: api.AugmentationConfig[] = Array.from(enabledAugs).map((id) => ({
      type: id,
      params: augmentations.find((a) => a.id === id)?.params || {},
    }));

    try {
      const result = await api.startTraining({
        dataset: selectedBuiltInDataset,
        preset: selectedPreset,
        epochs,
        batch_size: batchSize,
        optimizer: { type: selectedOptimizer, params: optimizerParams },
        scheduler: { type: selectedScheduler, params: schedulerParams },
        augmentations: augConfigs,
        name: modelName || undefined,
      });
      setTrainingJob({
        job_id: result.job_id,
        model_id: result.model_id,
        status: 'pending',
        epoch: 0,
        total_epochs: epochs,
        loss: 0,
        accuracy: 0,
        message: 'Starting...',
      });
    } catch (e) {
      setError('Failed to start training');
      setTraining(false);
    }
  }

  async function handleCancelTraining() {
    if (!trainingJob) return;
    try {
      await api.cancelTraining(trainingJob.job_id);
      setTraining(false);
      setTrainingJob(null);
    } catch (e) {
      setError('Failed to cancel training');
    }
  }

  const selectedBuiltInDatasetInfo = builtInDatasets.find((d) => d.id === selectedBuiltInDataset);
  const selectedPresetInfo = presets.find((p) => p.id === selectedPreset);

  const totalEpochs = mode === 'custom'
    ? architecture?.training?.epochs || 10
    : epochs;

  const currentEpoch = trainingJob
    ? ('epoch' in trainingJob ? (trainingJob.epoch ?? 0) : 0)
    : 0;

  return (
    <div className="train-tab">
      <div className="train-header">
        <div>
          <h2>Train a Model</h2>
          <p className="subtitle">Use presets for quick training or design a custom architecture with AI.</p>
        </div>
        {mode === 'custom' && (
          <button
            className={`btn helper-toggle ${helperOpen ? 'active' : ''}`}
            onClick={() => onHelperToggle?.(!helperOpen)}
          >
            {helperOpen ? 'Close Helper' : 'AI Helper'}
          </button>
        )}
      </div>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === 'quick' ? 'active' : ''}`}
          onClick={() => setMode('quick')}
          disabled={training}
        >
          Quick Start
        </button>
        <button
          className={`mode-btn ${mode === 'custom' ? 'active' : ''}`}
          onClick={() => setMode('custom')}
          disabled={training}
        >
          Custom Design
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {/* Quick Start Mode */}
      {mode === 'quick' && (
        <div className="quick-start-mode">
          <div className="form-group">
            <label>Dataset</label>
            <select
              value={selectedBuiltInDataset}
              onChange={(e) => setSelectedBuiltInDataset(e.target.value)}
              disabled={training}
            >
              {builtInDatasets.map((d) => (
                <option key={d.id} value={d.id} disabled={!d.available}>
                  {d.name} {!d.available && '(not available)'}
                </option>
              ))}
            </select>
            {selectedBuiltInDatasetInfo && (
              <p className="help-text">{selectedBuiltInDatasetInfo.description}</p>
            )}
          </div>

          <div className="form-group">
            <label>Architecture</label>
            <select
              value={selectedPreset}
              onChange={(e) => setSelectedPreset(e.target.value)}
              disabled={training}
            >
              {filteredPresets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            {selectedPresetInfo && <p className="help-text">{selectedPresetInfo.num_layers} layers</p>}
          </div>

          <div className="form-row">
            <div className="form-group half">
              <label>Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                min={1}
                max={100}
                disabled={training}
              />
            </div>
            <div className="form-group half">
              <label>Batch Size</label>
              <select value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))} disabled={training}>
                <option value={32}>32</option>
                <option value={64}>64</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Model Name (optional)</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="my_model"
              disabled={training}
            />
          </div>

          {/* Advanced Settings Toggle */}
          <button className="btn-link" onClick={() => setShowAdvanced(!showAdvanced)} disabled={training}>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>

          {showAdvanced && (
            <div className="advanced-settings">
              {/* Optimizer */}
              <div className="settings-section">
                <h4>Optimizer</h4>
                <div className="form-row">
                  <div className="form-group half">
                    <label>Type</label>
                    <select
                      value={selectedOptimizer}
                      onChange={(e) => setSelectedOptimizer(e.target.value)}
                      disabled={training}
                    >
                      {optimizers.map((o) => (
                        <option key={o.id} value={o.id}>
                          {o.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group half">
                    <label>Learning Rate</label>
                    <input
                      type="number"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                      step={0.001}
                      min={0.0001}
                      max={1}
                      disabled={training}
                    />
                  </div>
                </div>
                {selectedOptimizer === 'sgd' && (
                  <div className="form-row">
                    <div className="form-group half">
                      <label>Momentum</label>
                      <input
                        type="number"
                        value={momentum}
                        onChange={(e) => setMomentum(parseFloat(e.target.value) || 0.9)}
                        step={0.1}
                        min={0}
                        max={1}
                        disabled={training}
                      />
                    </div>
                    <div className="form-group half">
                      <label>Weight Decay</label>
                      <input
                        type="number"
                        value={weightDecay}
                        onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0)}
                        step={0.0001}
                        min={0}
                        max={0.1}
                        disabled={training}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Scheduler */}
              <div className="settings-section">
                <h4>Learning Rate Scheduler</h4>
                <div className="form-group">
                  <label>Type</label>
                  <select
                    value={selectedScheduler}
                    onChange={(e) => setSelectedScheduler(e.target.value)}
                    disabled={training}
                  >
                    {schedulers.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                </div>
                {selectedScheduler === 'step' && (
                  <div className="form-row">
                    <div className="form-group half">
                      <label>Step Size</label>
                      <input
                        type="number"
                        value={schedulerStepSize}
                        onChange={(e) => setSchedulerStepSize(parseInt(e.target.value) || 10)}
                        min={1}
                        max={50}
                        disabled={training}
                      />
                    </div>
                    <div className="form-group half">
                      <label>Gamma</label>
                      <input
                        type="number"
                        value={schedulerGamma}
                        onChange={(e) => setSchedulerGamma(parseFloat(e.target.value) || 0.1)}
                        step={0.1}
                        min={0.01}
                        max={1}
                        disabled={training}
                      />
                    </div>
                  </div>
                )}
                {selectedScheduler === 'exponential' && (
                  <div className="form-group">
                    <label>Gamma (decay per epoch)</label>
                    <input
                      type="number"
                      value={schedulerGamma}
                      onChange={(e) => setSchedulerGamma(parseFloat(e.target.value) || 0.95)}
                      step={0.01}
                      min={0.5}
                      max={0.99}
                      disabled={training}
                    />
                  </div>
                )}
              </div>

              {/* Data Augmentation */}
              <div className="settings-section">
                <h4>Data Augmentation</h4>
                <div className="aug-grid">
                  {augmentations.map((a) => (
                    <label key={a.id} className="aug-item">
                      <input
                        type="checkbox"
                        checked={enabledAugs.has(a.id)}
                        onChange={() => toggleAugmentation(a.id)}
                        disabled={training}
                      />
                      <span>{a.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Train Button */}
          {!training ? (
            <button className="btn primary large" onClick={handleTrainQuickStart}>
              Start Training
            </button>
          ) : (
            <button className="btn danger large" onClick={handleCancelTraining}>
              Cancel Training
            </button>
          )}
        </div>
      )}

      {/* Custom Design Mode */}
      {mode === 'custom' && (
        <div className="custom-design-mode">
          <div className="form-group">
            <label>Dataset</label>
            <select
              value={selectedDatasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              disabled={generating || training}
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
            {currentDataset && (
              <p className="help-text">
                Input: [{currentDataset.input_shape.join('x')}], {currentDataset.total_samples} samples
              </p>
            )}
          </div>

          <div className="form-group">
            <label>Describe Your Model</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., I want a CNN with good accuracy for image classification. Use batch normalization and dropout for regularization."
              rows={4}
              disabled={generating || training}
            />
          </div>

          <button
            className="btn primary"
            onClick={handleGenerate}
            disabled={!selectedDatasetId || !prompt.trim() || generating}
          >
            {generating ? 'Generating...' : 'Generate Architecture'}
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
                        <p key={i} className="validation-error">
                          {err}
                        </p>
                      ))}
                    </>
                  )}
                  {validation.warnings.map((warn, i) => (
                    <p key={i} className="validation-warning">
                      {warn}
                    </p>
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
                <button className="btn" onClick={handleRefine} disabled={!feedback.trim() || refining}>
                  {refining ? 'Refining...' : 'Refine'}
                </button>
              </div>

              {/* Train Button */}
              <div className="train-section">
                {!training ? (
                  <button
                    className="btn primary large"
                    onClick={handleTrainCustom}
                    disabled={validation !== null && !validation.valid}
                  >
                    Start Training
                  </button>
                ) : (
                  <button className="btn danger large" onClick={handleCancelTraining}>
                    Cancel Training
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Status (Shared) */}
      {trainingJob && (
        <div className="training-status">
          <h3>Training Progress</h3>
          <div className="status-grid">
            <div className="status-item">
              <span className="label">Status</span>
              <span className={`value status-${trainingJob.status}`}>{trainingJob.status}</span>
            </div>
            <div className="status-item">
              <span className="label">Epoch</span>
              <span className="value">
                {currentEpoch} / {totalEpochs}
              </span>
            </div>
            {'loss' in trainingJob && (
              <div className="status-item">
                <span className="label">Loss</span>
                <span className="value">{(trainingJob.loss || 0).toFixed(4)}</span>
              </div>
            )}
            {'accuracy' in trainingJob && (
              <div className="status-item">
                <span className="label">Accuracy</span>
                <span className="value">{(trainingJob.accuracy || 0).toFixed(2)}%</span>
              </div>
            )}
          </div>
          <p className="message">{trainingJob.message}</p>
          {['training', 'running'].includes(trainingJob.status) && currentEpoch > 0 && (
            <div className="progress-bar">
              <div className="progress" style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }} />
            </div>
          )}
          {trainingHistory.length > 0 && <TrainingChart data={trainingHistory} />}
          {trainingJob.status === 'completed' && (
            <div className="success-message">
              <p>
                Training completed! Model ID: <code>{trainingJob.model_id}</code>
              </p>
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
