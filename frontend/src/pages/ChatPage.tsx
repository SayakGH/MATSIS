import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useStore } from '../stores/useStore';
import { Send, Bot, User, Brain, Activity, ChevronDown, ChevronUp, Sparkles, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/query`;

export const ChatPage: React.FC = () => {
  const { dataset, sessionId, messages, addMessage, updateLastMessage, setIsStreaming, isStreaming } = useStore();
  const [input, setInput] = useState('');
  const [trace, setTrace] = useState<any[]>([]);
  const [showTrace, setShowTrace] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  // Use a ref to accumulate streaming tokens to avoid stale closure
  const streamBufferRef = useRef('');

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, trace]);

  const getSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        return resolve(socketRef.current);
      }

      const socket = new WebSocket(`${WS_URL}/${sessionId}`);
      socketRef.current = socket;

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.event === 'agent_start') {
          setTrace((prev) => {
            if (prev.find((s) => s.agent === data.agent && s.status === 'running')) return prev;
            return [...prev, { agent: data.agent, task: data.task, status: 'running' }];
          });
        } else if (data.event === 'agent_done') {
          setTrace((prev) =>
            prev.map((t) => (t.agent === data.agent ? { ...t, status: 'completed', output: data.output } : t))
          );
        } else if (data.event === 'stream_token') {
          streamBufferRef.current += data.token;
          updateLastMessage({ content: streamBufferRef.current });
        } else if (data.event === 'complete') {
          setIsStreaming(false);
          updateLastMessage({ chartData: data.final?.tool_result?.chart_data });
        } else if (data.event === 'error') {
          setIsStreaming(false);
          updateLastMessage({ content: `⚠️ Error: ${data.message}` });
        }
      };

      socket.onclose = () => {
        socketRef.current = null;
      };

      socket.onerror = () => {
        socketRef.current = null;
        reject(new Error('WebSocket connection failed'));
      };

      socket.onopen = () => resolve(socket);
    });
  }, [sessionId, updateLastMessage, setIsStreaming]);

  const handleSend = async () => {
    if (!input.trim() || !dataset || isStreaming) return;

    const userMsg = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: input,
      timestamp: new Date().toISOString(),
    };
    addMessage(userMsg);

    const botMsg = {
      id: (Date.now() + 1).toString(),
      role: 'assistant' as const,
      content: '',
      timestamp: new Date().toISOString(),
    };
    addMessage(botMsg);

    setIsStreaming(true);
    setTrace([]);
    streamBufferRef.current = '';

    const currentInput = input;
    setInput('');

    try {
      const socket = await getSocket();
      socket.send(JSON.stringify({ dataset_id: dataset.dataset_id, query: currentInput }));
    } catch {
      setIsStreaming(false);
      updateLastMessage({ content: '⚠️ Could not connect to backend. Is it running?' });
    }
  };

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto">
      {/* Trace Header */}
      {(isStreaming || trace.length > 0) && (
        <div className="mb-4 glass rounded-xl border border-indigo-500/20 p-4 transition-all animate-in slide-in-from-top-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400 animate-pulse" />
              <span className="text-sm font-semibold uppercase tracking-wider text-slate-400">Agent Pipeline</span>
            </div>
            <button onClick={() => setShowTrace(!showTrace)} className="text-slate-500 hover:text-slate-300">
              {showTrace ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>

          {showTrace && (
            <div className="flex gap-4 overflow-x-auto pb-2">
              {trace.map((step, i) => (
                <div
                  key={i}
                  className={cn(
                    'flex items-center gap-3 px-4 py-2 rounded-lg border min-w-[180px] shrink-0 transition-all',
                    step.status === 'running'
                      ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400'
                      : 'bg-slate-800/50 border-slate-700 text-slate-400'
                  )}
                >
                  {step.status === 'running' ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Brain className="w-4 h-4 text-indigo-500" />
                  )}
                  <div className="text-xs">
                    <p className="font-bold uppercase tracking-tighter">{step.agent}</p>
                    <p className="opacity-70 truncate max-w-[120px]">{step.task || 'working...'}</p>
                  </div>
                </div>
              ))}
              {trace.length === 0 && (
                <p className="text-xs text-slate-500 italic">Decomposing query into execution plan...</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-6 pb-28">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-40 py-20">
            <div className="w-20 h-20 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center">
              <Bot className="w-10 h-10" />
            </div>
            <div className="max-w-xs">
              <h3 className="font-semibold text-lg">Brain Ready</h3>
              <p className="text-sm">Ask me to forecast, detect anomalies, or summarize your dataset trends.</p>
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={cn(
              'flex gap-4 p-4 rounded-2xl transition-all',
              m.role === 'user' ? 'ml-12 bg-slate-800/30 border border-slate-800/50' : 'mr-12 glass'
            )}
          >
            <div
              className={cn(
                'w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-lg',
                m.role === 'user' ? 'bg-slate-700' : 'bg-gradient-to-br from-indigo-500 to-purple-600'
              )}
            >
              {m.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Sparkles className="w-5 h-5 text-white" />}
            </div>
            <div className="space-y-2 flex-1 overflow-hidden">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-widest text-slate-500">
                  {m.role === 'user' ? 'Human' : 'Agent Core'}
                </span>
                <span className="text-[10px] text-slate-600">{new Date(m.timestamp).toLocaleTimeString()}</span>
              </div>
              <div className="text-slate-200 leading-relaxed whitespace-pre-wrap min-h-[20px]">
                {m.content ? (
                  m.content
                ) : isStreaming && m.role === 'assistant' ? (
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                ) : null}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="sticky bottom-0 py-4 bg-gradient-to-t from-slate-950 via-slate-950/95 to-transparent">
        <div className="relative group max-w-5xl mx-auto">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-20 group-focus-within:opacity-40 transition duration-1000"></div>
          <div className="relative flex bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden focus-within:border-slate-700 transition-all">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder={dataset ? 'Ask about your time series data...' : 'Please upload a dataset first'}
              disabled={!dataset || isStreaming}
              className="flex-1 bg-transparent px-6 py-4 outline-none text-slate-200 disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || !dataset || isStreaming}
              className="px-6 flex items-center justify-center bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 transition-all"
            >
              {isStreaming ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>
        {!dataset && (
          <p className="text-center text-xs text-slate-500 mt-2">← Go to Upload to start</p>
        )}
      </div>
    </div>
  );
};
