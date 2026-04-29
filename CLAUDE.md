<!-- memoirs:start -->
## Memoirs (memory engine)

This workspace has the **Memoirs** MCP server available. Use it proactively.

### When to call `mcp_get_context`

- **Main / interactive session:** call it at task start with a query summarizing the goal. The result reduces history to relevant past decisions/preferences/facts in ~600-1500 tokens.
- **Sub-agents launched by the Agent / Task tool:** **DO NOT call** `mcp_get_context` or any other Memoirs MCP tool. The parent has already loaded context and provides it via the prompt. The Memoirs MCP server is single-instance stdio and parallel sub-agent calls cause stalls. Just go straight to the code.
- **Background / parallel jobs:** same rule — skip MCP, use the prompt context.

### Other MCP tools (main session only)

1. **When detecting a project / repo:** call `mcp_list_projects` and `mcp_get_project_context`.

2. **At task end (or whenever the user states a durable preference, decision, or constraint):** call `mcp_add_memory` with `type` ∈ {preference, fact, project, task, decision, style, credential_pointer}, `importance: 1..5, confidence: 0..1`.

3. **When you correct yourself:** call `mcp_score_feedback` with `useful=false` on the stale memory and add the corrected version with `mcp_add_memory`.

4. **For long conversations (50+ messages):** call `mcp_summarize_thread`.

Memory types: `preference, fact, project, task, decision, style, credential_pointer`.

<!-- memoirs:end -->
