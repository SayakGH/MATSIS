export interface DatasetMeta {
  dataset_id: string;
  filename: string;
  timestamp_col: string;
  value_cols: string[];
  row_count: number;
  date_range: [string, string];
}

export interface PlanStep {
  agent: string;
  task: string;
  params: Record<string, any>;
}

export interface PlanSchema {
  intent: string;
  steps: PlanStep[];
  target_column?: string;
  time_window?: string;
}

export interface AgentStep {
  agent: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  output?: any;
  task?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  agentTrace?: AgentStep[];
  chartData?: any[];
}
