import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react';
import axios from 'axios';
import { useStore } from '../stores/useStore';
import { useNavigate } from 'react-router-dom';
import { cn } from '../lib/utils';

export const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const setDataset = useStore((state) => state.setDataset);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData);
      setUploadResult(response.data);
      setDataset(response.data);
    } catch (error) {
      console.error('Upload failed', error);
      alert('Upload failed. Is the backend running?');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Import Dataset</h2>
        <p className="text-slate-400">Upload your CSV file for multi-agent time series analysis</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div 
          {...getRootProps()} 
          className={cn(
            "aspect-square rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-12 transition-all cursor-pointer",
            isDragActive ? "border-indigo-500 bg-indigo-500/5" : "border-slate-800 hover:border-slate-700 bg-slate-900/50"
          )}
        >
          <input {...getInputProps()} />
          <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-6">
            <Upload className={cn("w-8 h-8", isDragActive ? "text-indigo-400" : "text-slate-400")} />
          </div>
          <p className="font-medium text-lg mb-2">
            {file ? file.name : "Drag & drop CSV here"}
          </p>
          <p className="text-slate-500 text-sm">or click to browse files</p>
        </div>

        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 h-full flex flex-col">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-indigo-400" />
              Requirements
            </h3>
            <ul className="space-y-4 text-slate-400 text-sm flex-1">
              <li className="flex gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                Must be a valid CSV file
              </li>
              <li className="flex gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                First column should be a datetime format
              </li>
              <li className="flex gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                At least one numerical value column
              </li>
              <li className="flex gap-3 text-slate-500">
                <CheckCircle2 className="w-4 h-4 text-slate-700 shrink-0 mt-0.5" />
                Max file size: 50MB (Recommended for local)
              </li>
            </ul>

            <button
              onClick={handleUpload}
              disabled={!file || isUploading || uploadResult}
              className={cn(
                "w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all mt-8",
                !file || isUploading || uploadResult
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                  : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
              )}
            >
              {isUploading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Process Dataset"}
              {!isUploading && !uploadResult && <ArrowRight className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {uploadResult && (
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 animate-in zoom-in-95 duration-300">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-emerald-400 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5" />
              Dataset Ready
            </h3>
            <button 
              onClick={() => navigate('/chat')}
              className="text-indigo-400 hover:text-indigo-300 text-sm font-medium"
            >
              Start Chatting →
            </button>
          </div>
          
          <div className="grid grid-cols-3 gap-6">
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Rows</p>
              <p className="text-xl font-bold">{uploadResult.row_count}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Value Cols</p>
              <p className="text-xl font-bold">{uploadResult.detected_value_cols.length}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Time Range</p>
              <p className="text-sm font-bold truncate">Detected</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
