<!-- memoirs:start -->
## Memoirs (memory engine)

This workspace has the **Memoirs** MCP server available. Use it proactively:

1. **At task start:** call `mcp_get_context` with a query summarizing the goal.
   The result reduces the long history to relevant past decisions, preferences,
   and project facts in ~600-1500 tokens — feed it into your reasoning.

2. **When detecting a project / repo:** call `mcp_list_projects` and
   `mcp_get_project_context` for that project to load its scoped memory.

3. **At task end (or whenever the user states a durable preference, decision, or
   constraint):** call `mcp_add_memory` with `type` ∈
   {preference, fact, project, task, decision, style, credential_pointer},
   `importance: 1..5, confidence: 0..1`. Aim for type=preference / decision /
   project — avoid type=fact for generic statements.

4. **When you correct yourself or invalidate an earlier statement:** call
   `mcp_score_feedback` with `useful=false` on the stale memory and add the
   corrected version with `mcp_add_memory`.

5. **For long conversations (50+ messages):** call `mcp_summarize_thread` to
   compress them into a single durable memory.

Memory types: `preference, fact, project, task, decision, style, credential_pointer`.

<!-- memoirs:end -->
