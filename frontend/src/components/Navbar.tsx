import React from 'react';
import { useStore } from '../stores/useStore';
import { Database, Activity } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { dataset } = useStore();

  return (
    <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 glass z-10">
      <div className="flex items-center gap-2">
        <Activity className="w-6 h-6 text-indigo-500" />
        <h1 className="text-xl font-bold tracking-tight">MATSIS <span className="text-xs font-normal text-slate-500 ml-2">v1.0</span></h1>
      </div>
      
      <div className="flex items-center gap-4">
        {dataset && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-sm">
            <Database className="w-4 h-4 text-indigo-400" />
            <span className="max-w-[150px] truncate">{dataset.filename}</span>
          </div>
        )}
      </div>
    </header>
  );
};
