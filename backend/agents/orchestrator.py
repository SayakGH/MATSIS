import asyncio
import time
import uuid
import pandas as pd
from typing import AsyncGenerator

from agents.planner import planner_agent
from agents.tool_agent import tool_agent
from agents.analyst import analyst_agent
from agents.explainer import explainer_agent
from agents.memory import memory_agent
from agents.preprocessor import preprocessor_agent
from agents.postprocessor import postprocessor_agent
from models.schemas import QueryRecord, DatasetMeta, ToolResult


class Orchestrator:
    async def execute_query(
        self, query: str, dataset_meta: DatasetMeta, session_id: str, df: pd.DataFrame
    ) -> AsyncGenerator:
        start_time = time.time()
        query_id = str(uuid.uuid4())

        # ── Stage 1: preprocess + plan in parallel ────────────────────────────
        # preprocessor is CPU-bound so run in a thread; planner is async/IO
        yield {"event": "agent_start", "agent": "preprocessor"}
        yield {"event": "agent_start", "agent": "planner"}

        try:
            df_clean, plan = await asyncio.gather(
                asyncio.to_thread(preprocessor_agent.clean, df),
                planner_agent.plan(query, dataset_meta),
            )
        except Exception as e:
            yield {"event": "error", "message": f"Stage 1 failed: {e}"}
            return

        yield {"event": "agent_done", "agent": "preprocessor"}
        yield {"event": "agent_done", "agent": "planner", "output": plan.model_dump()}

        # ── Stage 2: tool execution (sequential — needs clean df + plan) ──────
        tool_result: ToolResult | None = None
        analysis: dict | None = None

        for step in plan.steps:
            yield {"event": "agent_start", "agent": step.agent, "task": step.task}

            if step.agent == "tool":
                tool_result = tool_agent.run(step.task, step.params, df_clean)
                yield {"event": "agent_done", "agent": "tool", "output": tool_result.model_dump()}
                if tool_result.error:
                    yield {"event": "error", "message": tool_result.error}
                    return

            elif step.agent == "analyst":
                if tool_result:
                    # Stage 3a: run analyst + prefetch context in parallel
                    analysis, _ = await asyncio.gather(
                        analyst_agent.analyze(query, tool_result, session_id),
                        memory_agent.get_context_prompt(session_id),  # warm cache
                    )
                    analysis = postprocessor_agent.refine(
                        analysis, tool_result.metrics if tool_result else None
                    )
                    yield {"event": "agent_done", "agent": "analyst", "output": analysis}
                else:
                    yield {"event": "agent_error", "agent": "analyst", "error": "No tool result to analyse"}

            elif step.agent == "explainer":
                if analysis:
                    yield {"event": "agent_streaming", "agent": "explainer"}
                    full_explanation = ""
                    async for token in explainer_agent.explain_stream(query, analysis, session_id):
                        full_explanation += token
                        yield {"event": "stream_token", "token": token}

                    # Persist record with follow-up suggestions
                    latency = int((time.time() - start_time) * 1000)
                    follow_ups = analysis.get("follow_up_suggestions", [])
                    record = QueryRecord(
                        query_id=query_id,
                        session_id=session_id,
                        raw_query=query,
                        plan=plan.model_dump(),
                        tool_result=tool_result.model_dump() if tool_result else None,
                        analysis=analysis,
                        explanation=full_explanation,
                        follow_up_suggestions=follow_ups,
                        latency_ms=latency,
                    )
                    await memory_agent.store_query(session_id, record)
                    yield {"event": "complete", "final": record.model_dump(mode="json")}
                else:
                    yield {"event": "agent_error", "agent": "explainer", "error": "No analysis to explain"}

            else:
                print(f"⚠️ Unknown agent in plan step: '{step.agent}' — skipping")
                yield {"event": "agent_done", "agent": step.agent, "output": "skipped"}


orchestrator = Orchestrator()
