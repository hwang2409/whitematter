import { useState, useEffect } from 'react';
import * as api from '../api';

export default function TrainTab() {
  const [datasets, setDatasets] = useState<api.Dataset[]>([]);
  const [presets, setPresets] = useState<api.Preset[]>([]);
  const [optimizers, setOptimizers] = useState<api.Optimizer[]>([]);
  const [schedulers, setSchedulers] = useState<api.Scheduler[]>([]);
  const [augmentations, setAugmentations] = useState<api.Augmentation[]>([]);

  // Form state
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(64);
  const [modelName, setModelName] = useState('');

  // Optimizer config
  const [selectedOptimizer, setSelectedOptimizer] = useState('sgd');
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.0);

  // Scheduler config
  const [selectedScheduler, setSelectedScheduler] = useState('none');
  const [schedulerStepSize, setSchedulerStepSize] = useState(10);
  const [schedulerGamma, setSchedulerGamma] = useState(0.1);

  // Augmentation config
  const [enabledAugs, setEnabledAugs] = useState<Set<string>>(new Set());

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [currentJob, setCurrentJob] = useState<api.TrainJob | null>(null);
  const [error, setError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    loadOptions();
  }, []);

  useEffect(() => {
    let interval: number;
    if (currentJob && ['pending', 'running'].includes(currentJob.status)) {
      interval = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus(currentJob.id);
          setCurrentJob(status);
          if (!['pending', 'running'].includes(status.status)) {
            setIsTraining(false);
          }
        } catch (e) {
          console.error('Failed to get training status:', e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [currentJob?.id, currentJob?.status]);

  async function loadOptions() {
    try {
      const [ds, ps, opts, scheds, augs] = await Promise.all([
        api.getDatasets(),
        api.getPresets(),
        api.getOptimizers(),
        api.getSchedulers(),
        api.getAugmentations(),
      ]);
      setDatasets(ds);
      setPresets(ps);
      setOptimizers(opts);
      setSchedulers(scheds);
      setAugmentations(augs);

      if (ds.length > 0) setSelectedDataset(ds[0].id);
      const filtered = ps.filter(p => p.dataset === ds[0]?.id);
      if (filtered.length > 0) setSelectedPreset(filtered[0].id);
    } catch (e) {
      setError('Failed to load options');
    }
  }

  // Filter presets for selected dataset
  const filteredPresets = presets.filter(p => p.dataset === selectedDataset);

  // Update preset when dataset changes
  useEffect(() => {
    const filtered = presets.filter(p => p.dataset === selectedDataset);
    if (filtered.length > 0 && !filtered.find(p => p.id === selectedPreset)) {
      setSelectedPreset(filtered[0].id);
    }
  }, [selectedDataset, presets]);

  function toggleAugmentation(id: string) {
    const newSet = new Set(enabledAugs);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setEnabledAugs(newSet);
  }

  async function handleStartTraining() {
    if (!selectedDataset || !selectedPreset) {
      setError('Please select dataset and architecture');
      return;
    }

    setError('');
    setIsTraining(true);

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

    const augConfigs: api.AugmentationConfig[] = Array.from(enabledAugs).map(id => ({
      type: id,
      params: augmentations.find(a => a.id === id)?.params || {},
    }));

    try {
      const result = await api.startTraining({
        dataset: selectedDataset,
        preset: selectedPreset,
        epochs,
        batch_size: batchSize,
        optimizer: { type: selectedOptimizer, params: optimizerParams },
        scheduler: { type: selectedScheduler, params: schedulerParams },
        augmentations: augConfigs,
        name: modelName || undefined,
      });
      setCurrentJob({
        id: result.job_id,
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
      setIsTraining(false);
    }
  }

  async function handleCancelTraining() {
    if (currentJob) {
      try {
        await api.cancelTraining(currentJob.id);
        setIsTraining(false);
        setCurrentJob(null);
      } catch (e) {
        setError('Failed to cancel training');
      }
    }
  }

  const selectedDatasetInfo = datasets.find(d => d.id === selectedDataset);
  const selectedPresetInfo = presets.find(p => p.id === selectedPreset);

  return (
    <div className="train-tab">
      <h2>Train a New Model</h2>

      {error && <div className="error">{error}</div>}

      {/* Basic Settings */}
      <div className="form-group">
        <label>Dataset</label>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          disabled={isTraining}
        >
          {datasets.map(d => (
            <option key={d.id} value={d.id} disabled={!d.available}>
              {d.name} {!d.available && '(not available)'}
            </option>
          ))}
        </select>
        {selectedDatasetInfo && (
          <p className="help-text">{selectedDatasetInfo.description}</p>
        )}
      </div>

      <div className="form-group">
        <label>Architecture</label>
        <select
          value={selectedPreset}
          onChange={(e) => setSelectedPreset(e.target.value)}
          disabled={isTraining}
        >
          {filteredPresets.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        {selectedPresetInfo && (
          <p className="help-text">{selectedPresetInfo.num_layers} layers</p>
        )}
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
            disabled={isTraining}
          />
        </div>
        <div className="form-group half">
          <label>Batch Size</label>
          <select value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))} disabled={isTraining}>
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
          disabled={isTraining}
        />
      </div>

      {/* Advanced Settings Toggle */}
      <button
        className="btn-link"
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={isTraining}
      >
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
                  disabled={isTraining}
                >
                  {optimizers.map(o => (
                    <option key={o.id} value={o.id}>{o.name}</option>
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
                  disabled={isTraining}
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
                    disabled={isTraining}
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
                    disabled={isTraining}
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
                disabled={isTraining}
              >
                {schedulers.map(s => (
                  <option key={s.id} value={s.id}>{s.name}</option>
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
                    disabled={isTraining}
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
                    disabled={isTraining}
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
                  disabled={isTraining}
                />
              </div>
            )}
          </div>

          {/* Data Augmentation */}
          <div className="settings-section">
            <h4>Data Augmentation</h4>
            <div className="aug-grid">
              {augmentations.map(a => (
                <label key={a.id} className="aug-item">
                  <input
                    type="checkbox"
                    checked={enabledAugs.has(a.id)}
                    onChange={() => toggleAugmentation(a.id)}
                    disabled={isTraining}
                  />
                  <span>{a.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      {!isTraining ? (
        <button className="btn primary" onClick={handleStartTraining}>
          Start Training
        </button>
      ) : (
        <button className="btn danger" onClick={handleCancelTraining}>
          Cancel Training
        </button>
      )}

      {/* Training Status */}
      {currentJob && (
        <div className="training-status">
          <h3>Training Progress</h3>
          <div className="status-grid">
            <div className="status-item">
              <span className="label">Status</span>
              <span className={`value status-${currentJob.status}`}>{currentJob.status}</span>
            </div>
            <div className="status-item">
              <span className="label">Epoch</span>
              <span className="value">{currentJob.epoch} / {currentJob.total_epochs || epochs}</span>
            </div>
            <div className="status-item">
              <span className="label">Loss</span>
              <span className="value">{currentJob.loss.toFixed(4)}</span>
            </div>
            <div className="status-item">
              <span className="label">Accuracy</span>
              <span className="value">{currentJob.accuracy.toFixed(2)}%</span>
            </div>
          </div>
          <p className="message">{currentJob.message}</p>
          {currentJob.status === 'running' && (
            <div className="progress-bar">
              <div
                className="progress"
                style={{ width: `${(currentJob.epoch / (currentJob.total_epochs || epochs)) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
