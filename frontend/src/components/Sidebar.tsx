import { NavLink } from 'react-router-dom';
import { LayoutDashboard, MessageSquare, Upload, Settings } from 'lucide-react';
import { cn } from '../lib/utils';
import { useStore } from '../stores/useStore';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
  { icon: Upload, label: 'Upload', path: '/upload' },
  { icon: MessageSquare, label: 'Chat', path: '/chat' },
];

export const Sidebar: React.FC = () => {
  const { dataset } = useStore();

  return (
    <aside className="w-64 border-r border-slate-800 flex flex-col glass shrink-0">
      <div className="p-6 flex flex-col h-full">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-white text-sm">M</div>
          <div>
            <span className="font-semibold text-lg block leading-none">MATSIS</span>
            <span className="text-[10px] text-slate-500 uppercase tracking-widest">v1.0</span>
          </div>
        </div>

        {/* Nav */}
        <nav className="space-y-1 flex-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 text-sm",
                isActive
                  ? "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              )}
            >
              <item.icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Active dataset indicator */}
        {dataset && (
          <div className="mt-4 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-[10px] text-emerald-400 uppercase tracking-widest mb-1">Active dataset</p>
            <p className="text-xs text-slate-300 truncate font-medium">{dataset.filename}</p>
            <p className="text-[10px] text-slate-500">{dataset.row_count?.toLocaleString()} rows</p>
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-slate-800">
          <button className="flex items-center gap-3 text-slate-400 hover:text-slate-200 transition-colors w-full px-3 py-2 rounded-lg hover:bg-slate-800 text-sm">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
