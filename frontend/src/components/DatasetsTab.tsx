import { useState, useEffect, useRef } from 'react';
import * as api from '../api';
import ConfirmDialog from './ConfirmDialog';

interface Props {
  onDatasetSelect?: (id: string | null) => void;
  onUpdate?: () => void;
}

export default function DatasetsTab({ onDatasetSelect, onUpdate }: Props) {
  const [datasets, setDatasets] = useState<api.CustomDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<api.CustomDataset | null>(null);
  const [preview, setPreview] = useState<api.DatasetPreviewSample[]>([]);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<api.CustomDataset | null>(null);
  const [uploadPreview, setUploadPreview] = useState<api.DatasetPreviewSample[]>([]);
  const [error, setError] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<'zip' | 'text'>('zip');
  const [tokenizerType, setTokenizerType] = useState<'character' | 'word'>('character');
  const [seqLength, setSeqLength] = useState(128);
  const [textPreview, setTextPreview] = useState<api.TextPreview | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Confirm dialog state
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadPreview(selectedDataset.id);
    } else {
      setPreview([]);
    }
  }, [selectedDataset?.id]);

  async function loadPreview(datasetId: string) {
    setLoadingPreview(true);
    try {
      const data = await api.getDatasetPreview(datasetId);
      setPreview(data.samples || []);
      // Handle text preview
      if ((data as any).text_preview) {
        setTextPreview((data as any).text_preview);
      } else {
        setTextPreview(null);
      }
    } catch (e) {
      console.error('Failed to load preview:', e);
      setPreview([]);
      setTextPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function loadDatasets() {
    setLoading(true);
    try {
      const data = await api.getCustomDatasets();
      setDatasets(data);
      onUpdate?.();
    } catch (e) {
      setError('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  }

  function handleSelectDataset(dataset: api.CustomDataset) {
    setSelectedDataset(dataset);
    onDatasetSelect?.(dataset.id);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError('');
      // Detect file type and auto-fill name
      if (f.name.endsWith('.txt')) {
        setFileType('text');
        if (!datasetName) {
          setDatasetName(f.name.replace(/\.txt$/i, ''));
        }
      } else if (f.name.endsWith('.zip')) {
        setFileType('zip');
        if (!datasetName) {
          setDatasetName(f.name.replace(/\.zip$/i, ''));
        }
      }
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith('.zip') || f.name.endsWith('.txt'))) {
      setFile(f);
      setError('');
      if (f.name.endsWith('.txt')) {
        setFileType('text');
        if (!datasetName) {
          setDatasetName(f.name.replace(/\.txt$/i, ''));
        }
      } else {
        setFileType('zip');
        if (!datasetName) {
          setDatasetName(f.name.replace(/\.zip$/i, ''));
        }
      }
    } else {
      setError('Please upload a ZIP or TXT file');
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  async function handleUpload() {
    if (!file) {
      setError('Please select a file');
      return;
    }
    if (!datasetName.trim()) {
      setError('Please enter a dataset name');
      return;
    }

    setUploading(true);
    setError('');

    try {
      let dataset: api.CustomDataset;
      if (fileType === 'text') {
        dataset = await api.uploadTextDataset(file, datasetName.trim(), {
          tokenizer_type: tokenizerType,
          seq_length: seqLength,
        });
      } else {
        dataset = await api.uploadDataset(file, datasetName.trim());
      }
      setDatasets((prev) => [dataset, ...prev]);
      setFile(null);
      setDatasetName('');
      setFileType('zip');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      // Show upload success with preview
      setUploadedDataset(dataset);
      setSelectedDataset(dataset);

      // Load preview for the uploaded dataset
      try {
        const previewData = await api.getDatasetPreview(dataset.id);
        setUploadPreview(previewData.samples || []);
        // For text datasets, also get text preview
        if (dataset.data_type === 'text' && (previewData as any).text_preview) {
          setTextPreview((previewData as any).text_preview);
        }
      } catch (e) {
        console.error('Failed to load upload preview:', e);
        setUploadPreview([]);
      }
    } catch (e: any) {
      setError(e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function confirmDelete(id: string) {
    try {
      await api.deleteCustomDataset(id);
      setDatasets(datasets.filter((d) => d.id !== id));
      if (selectedDataset?.id === id) {
        setSelectedDataset(null);
      }
    } catch (e) {
      setError('Failed to delete dataset');
    } finally {
      setDeleteConfirm(null);
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleString();
  }

  function getStatusClass(status: string) {
    switch (status) {
      case 'ready':
        return 'status-completed';
      case 'processing':
        return 'status-running';
      case 'error':
        return 'status-failed';
      default:
        return 'status-pending';
    }
  }

  if (loading) {
    return <div className="datasets-tab"><p>Loading datasets...</p></div>;
  }

  return (
    <div className="datasets-tab">
      <h2>Datasets</h2>

      {error && <div className="error">{error}</div>}

      {/* Upload Section */}
      <div className="upload-section">
        <h3>Upload New Dataset</h3>
        <p className="help-text">
          Upload a ZIP file for images, or a TXT file for text/language model training.
        </p>

        <div className="form-group">
          <label>Dataset Name</label>
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            placeholder="my_dataset"
            disabled={uploading}
          />
        </div>

        <div
          className={`drop-zone ${file ? 'has-file' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !file && fileInputRef.current?.click()}
        >
          {file ? (
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
              <span className="file-type">{fileType === 'text' ? 'Text' : 'Images'}</span>
              <button
                className="clear-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setFileType('zip');
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
              >
                &times;
              </button>
            </div>
          ) : (
            <div className="drop-zone-content">
              <p>Drop a file here or click to upload</p>
              <p className="hint">ZIP: folder-per-class, MNIST IDX | TXT: text for language models</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip,.txt"
            onChange={handleFileChange}
            hidden
          />
        </div>

        {/* Text-specific options */}
        {fileType === 'text' && file && (
          <div className="text-options">
            <div className="form-row">
              <div className="form-group">
                <label>Tokenizer</label>
                <select
                  value={tokenizerType}
                  onChange={(e) => setTokenizerType(e.target.value as 'character' | 'word')}
                  disabled={uploading}
                >
                  <option value="character">Character-level</option>
                  <option value="word">Word-level</option>
                </select>
              </div>
              <div className="form-group">
                <label>Sequence Length</label>
                <input
                  type="number"
                  value={seqLength}
                  onChange={(e) => setSeqLength(parseInt(e.target.value) || 128)}
                  min={32}
                  max={512}
                  disabled={uploading}
                />
              </div>
            </div>
            <p className="hint">
              Character-level works well for small datasets like Shakespeare.
              Word-level is better for larger corpora.
            </p>
          </div>
        )}

        <button
          className="btn primary"
          onClick={handleUpload}
          disabled={!file || !datasetName.trim() || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>

        {/* Upload Success Preview */}
        {uploadedDataset && (
          <div className="upload-success">
            <div className="upload-success-header">
              <h4>Upload Successful: {uploadedDataset.name}</h4>
              <button
                className="btn-close"
                onClick={() => {
                  setUploadedDataset(null);
                  setUploadPreview([]);
                }}
              >
                &times;
              </button>
            </div>
            <div className="upload-success-info">
              <span>{uploadedDataset.data_type}</span>
              <span>{uploadedDataset.format?.replace(/_/g, ' ')}</span>
              <span>{uploadedDataset.num_classes} classes</span>
              <span>{uploadedDataset.total_samples?.toLocaleString()} samples</span>
              {uploadedDataset.input_shape?.length > 0 && (
                <span>Shape: [{uploadedDataset.input_shape.join('x')}]</span>
              )}
            </div>
            {uploadedDataset.class_names?.length > 0 && (
              <div className="upload-success-classes">
                {uploadedDataset.class_names.map((name, i) => (
                  <span key={i} className="class-chip">{name}</span>
                ))}
              </div>
            )}
            {uploadedDataset.data_type === 'text' && textPreview ? (
              <div className="text-preview-section">
                <h5>Text Preview</h5>
                <div className="text-preview-stats">
                  <span>Vocab: {textPreview.vocab_size} tokens</span>
                  <span>Sequences: {textPreview.train_sequences} train / {textPreview.test_sequences} test</span>
                  <span>Total: {textPreview.total_tokens?.toLocaleString()} tokens</span>
                </div>
                {textPreview.sample_text && (
                  <div className="text-sample">
                    <pre>{textPreview.sample_text.slice(0, 300)}...</pre>
                  </div>
                )}
                {textPreview.sample_tokens?.length > 0 && (
                  <div className="token-preview">
                    <span className="label">Sample tokens:</span>
                    <div className="token-chips">
                      {textPreview.sample_tokens.slice(0, 20).map((t, i) => (
                        <span key={i} className="token-chip">{t === ' ' ? '␣' : t === '\n' ? '↵' : t}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : uploadPreview.length > 0 && (
              <div className="upload-preview">
                <h5>Sample Images</h5>
                <div className="preview-grid">
                  {uploadPreview.map((sample, i) => (
                    <div key={i} className="preview-item">
                      <img
                        src={`data:image/png;base64,${sample.image}`}
                        alt={sample.label}
                      />
                      <span className="preview-label">{sample.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Datasets List */}
      <div className="datasets-header">
        <h3>Your Datasets</h3>
        <button className="btn" onClick={loadDatasets}>Refresh</button>
      </div>

      {datasets.length === 0 ? (
        <p className="empty-state">No datasets yet. Upload one above!</p>
      ) : (
        <div className="datasets-layout">
          <div className="datasets-list">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`dataset-card ${selectedDataset?.id === dataset.id ? 'selected' : ''}`}
                onClick={() => handleSelectDataset(dataset)}
              >
                <div className="dataset-card-header">
                  <h4>{dataset.name}</h4>
                  <span className={`status-badge ${getStatusClass(dataset.status)}`}>
                    {dataset.status}
                  </span>
                </div>
                <div className="dataset-card-info">
                  <span>{dataset.data_type}</span>
                  <span>{dataset.num_classes} classes</span>
                  <span>{dataset.total_samples} samples</span>
                </div>
              </div>
            ))}
          </div>

          {selectedDataset && (
            <div className="dataset-details">
              <h3>{selectedDataset.name}</h3>

              <div className="details-grid">
                <div className="detail-item">
                  <span className="label">ID</span>
                  <span className="value">{selectedDataset.id}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Type</span>
                  <span className="value">{selectedDataset.data_type}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Format</span>
                  <span className="value">{selectedDataset.format?.replace(/_/g, ' ') || 'unknown'}</span>
                </div>
                {selectedDataset.input_shape?.length > 0 && (
                  <div className="detail-item">
                    <span className="label">Input Shape</span>
                    <span className="value">[{selectedDataset.input_shape.join('x')}]</span>
                  </div>
                )}
                <div className="detail-item">
                  <span className="label">Classes</span>
                  <span className="value">{selectedDataset.num_classes}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Total Samples</span>
                  <span className="value">{selectedDataset.total_samples.toLocaleString()}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Created</span>
                  <span className="value">{formatDate(selectedDataset.created_at)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Status</span>
                  <span className={`value ${getStatusClass(selectedDataset.status)}`}>
                    {selectedDataset.status}
                  </span>
                </div>
              </div>

              {selectedDataset.class_names?.length > 0 && (
                <div className="class-list">
                  <h4>Classes</h4>
                  <div className="class-chips">
                    {selectedDataset.class_names.map((name, i) => (
                      <span key={i} className="class-chip">{name}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="dataset-preview">
                <h4>Sample Data</h4>
                {loadingPreview ? (
                  <p className="loading-text">Loading preview...</p>
                ) : selectedDataset.data_type === 'text' && textPreview ? (
                  <div className="text-preview-section">
                    <div className="text-preview-stats">
                      <span>Vocab: {textPreview.vocab_size} tokens</span>
                      <span>Type: {textPreview.tokenizer_type}</span>
                      {textPreview.seq_length && <span>Seq length: {textPreview.seq_length}</span>}
                    </div>
                    {textPreview.sample_text && (
                      <div className="text-sample">
                        <pre>{textPreview.sample_text}</pre>
                      </div>
                    )}
                  </div>
                ) : preview.length > 0 ? (
                  <div className="preview-grid">
                    {preview.map((sample, i) => (
                      <div key={i} className="preview-item">
                        <img
                          src={`data:image/png;base64,${sample.image}`}
                          alt={sample.label}
                        />
                        <span className="preview-label">{sample.label}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-text">No preview available</p>
                )}
              </div>

              <div className="dataset-actions">
                <button
                  className="btn danger"
                  onClick={() => setDeleteConfirm(selectedDataset.id)}
                >
                  Delete Dataset
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <ConfirmDialog
        isOpen={deleteConfirm !== null}
        title="Delete Dataset"
        message="Are you sure you want to delete this dataset? This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={() => deleteConfirm && confirmDelete(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
      />
    </div>
  );
}
