import { create } from 'zustand';
import type { DatasetMeta, Message, AgentStep } from '../types';

interface AppState {
  dataset: DatasetMeta | null;
  sessionId: string;
  messages: Message[];
  currentTrace: AgentStep[];
  isStreaming: boolean;
  
  setDataset: (dataset: DatasetMeta | null) => void;
  addMessage: (message: Message) => void;
  updateLastMessage: (update: Partial<Message>) => void;
  setTrace: (trace: AgentStep[]) => void;
  updateTraceStep: (agent: string, status: AgentStep['status'], output?: any) => void;
  setIsStreaming: (isStreaming: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  dataset: null,
  sessionId: crypto.randomUUID(),
  messages: [],
  currentTrace: [],
  isStreaming: false,

  setDataset: (dataset) => set({ dataset }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  updateLastMessage: (update) => set((state) => {
    const newMessages = [...state.messages];
    if (newMessages.length > 0) {
      newMessages[newMessages.length - 1] = { ...newMessages[newMessages.length - 1], ...update };
    }
    return { messages: newMessages };
  }),
  setTrace: (trace) => set({ currentTrace: trace }),
  updateTraceStep: (agent, status, output) => set((state) => ({
    currentTrace: state.currentTrace.map((step) => 
      step.agent === agent ? { ...step, status, output: output || step.output } : step
    )
  })),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
}));
