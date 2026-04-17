import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../stores/useStore';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area,
} from 'recharts';
import {
  LayoutDashboard, Database, Calendar, Hash, TrendingUp,
  MessageSquare, Upload, RefreshCw, ChevronRight,
} from 'lucide-react';
import axios from 'axios';
import { cn } from '../lib/utils';

interface DatasetCard {
  dataset_id: string;
  filename: string;
  timestamp_col: string;
  value_cols: string[];
  row_count: number;
  date_range: string[];
  uploaded_at: string;
}

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset, setDataset } = useStore();
  const [datasets, setDatasets] = useState<DatasetCard[]>([]);
  const [preview, setPreview] = useState<any[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingPreview, setLoadingPreview] = useState(false);

  // ── Fetch dataset list ──────────────────────────────────────────────────────
  const fetchDatasets = async () => {
    setLoadingList(true);
    try {
      const res = await axios.get('/api/datasets');
      setDatasets(res.data);
    } catch (e) {
      console.error('Failed to fetch datasets', e);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => { fetchDatasets(); }, []);

  // ── Fetch preview when dataset changes ─────────────────────────────────────
  useEffect(() => {
    if (!dataset) { setPreview([]); return; }
    setLoadingPreview(true);
    axios.get(`/api/datasets/${dataset.dataset_id}/preview`)
      .then(r => setPreview(r.data.chart_data || []))
      .catch(() => setPreview([]))
      .finally(() => setLoadingPreview(false));
  }, [dataset?.dataset_id]);

  const handleSelect = (ds: DatasetCard) => {
    setDataset(ds as any);
  };

  const handleChat = () => navigate('/chat');

  // ── Empty state ─────────────────────────────────────────────────────────────
  if (!loadingList && datasets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-50">
        <Database className="w-16 h-16" />
        <h2 className="text-xl font-semibold">No datasets yet</h2>
        <p className="text-sm max-w-xs">Upload a CSV file to get started.</p>
        <button
          onClick={() => navigate('/upload')}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm transition-all"
        >
          <Upload className="w-4 h-4" /> Upload Dataset
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex gap-6 animate-in fade-in duration-300">

      {/* ── LEFT: Dataset list ──────────────────────────────────────────────── */}
      <div className="w-80 shrink-0 flex flex-col gap-3">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <LayoutDashboard className="w-4 h-4" /> Datasets
          </h2>
          <button
            onClick={fetchDatasets}
            className="text-slate-500 hover:text-slate-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", loadingList && "animate-spin")} />
          </button>
        </div>

        <div className="space-y-2 overflow-y-auto">
          {datasets.map((ds) => {
            const isActive = dataset?.dataset_id === ds.dataset_id;
            return (
              <button
                key={ds.dataset_id}
                onClick={() => handleSelect(ds)}
                className={cn(
                  "w-full text-left p-4 rounded-xl border transition-all duration-200 group",
                  isActive
                    ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-300"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/50 text-slate-300"
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="font-semibold text-sm truncate">{ds.filename}</p>
                    <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <Hash className="w-3 h-3" />{ds.row_count?.toLocaleString()} rows
                      </span>
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {ds.date_range?.[0]?.slice(0, 10)} →
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {ds.value_cols?.slice(0, 3).map(c => (
                        <span key={c} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                  <ChevronRight className={cn(
                    "w-4 h-4 shrink-0 mt-0.5 transition-transform",
                    isActive ? "text-indigo-400" : "text-slate-600 group-hover:translate-x-0.5"
                  )} />
                </div>
              </button>
            );
          })}
        </div>

        <button
          onClick={() => navigate('/upload')}
          className="mt-2 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-dashed border-slate-700 text-slate-500 hover:text-slate-300 hover:border-slate-600 transition-all text-sm"
        >
          <Upload className="w-4 h-4" /> Upload new dataset
        </button>
      </div>

      {/* ── RIGHT: Preview panel ────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        {!dataset ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center opacity-40 space-y-3">
            <TrendingUp className="w-12 h-12" />
            <p className="text-sm">Select a dataset to see its preview</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-bold">{dataset.filename}</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  {dataset.row_count?.toLocaleString()} rows · {dataset.value_cols?.join(', ')} · {dataset.timestamp_col}
                </p>
              </div>
              <button
                onClick={handleChat}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-sm font-medium transition-all shadow-lg shadow-indigo-500/20"
              >
                <MessageSquare className="w-4 h-4" />
                Chat with this data
              </button>
            </div>

            {/* Stat cards */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Rows", value: dataset.row_count?.toLocaleString() },
                { label: "From", value: dataset.date_range?.[0]?.slice(0, 10) },
                { label: "To", value: dataset.date_range?.[1]?.slice(0, 10) },
              ].map((s) => (
                <div key={s.label} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest">{s.label}</p>
                  <p className="text-sm font-semibold mt-1">{s.value ?? '—'}</p>
                </div>
              ))}
            </div>

            {/* Chart */}
            <div className="flex-1 min-h-0 p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                Time Series Preview ({dataset.value_cols?.[0]})
              </h4>
              {loadingPreview ? (
                <div className="flex items-center justify-center h-full text-slate-600 text-sm">Loading preview…</div>
              ) : preview.length === 0 ? (
                <div className="flex items-center justify-center h-full text-slate-600 text-sm">No data available</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={preview} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#475569"
                      fontSize={10}
                      tickFormatter={(v) => v?.slice(0, 10)}
                      interval="preserveStartEnd"
                    />
                    <YAxis stroke="#475569" fontSize={10} width={48} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: 12 }}
                      itemStyle={{ color: '#f8fafc' }}
                      labelFormatter={(v) => v?.slice(0, 10)}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#6366f1"
                      strokeWidth={2}
                      fill="url(#grad)"
                      dot={false}
                      name={dataset.value_cols?.[0]}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
