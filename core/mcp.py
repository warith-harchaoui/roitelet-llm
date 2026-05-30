"""Embedded MCP-style JSON-RPC handlers for Roitelet.

The implementation is intentionally compact and HTTP-friendly. It exposes one
main tool, `roitelet.chat`, allowing MCP-capable clients to call the router and
receive structured metadata about the selected models and final fused answer.
"""

from __future__ import annotations

from typing import Any

from .pipeline import run_roitelet_chat
from .schemas import ChatRequest, MCPRequest, RouterPreferences


async def handle_mcp_request(payload: MCPRequest) -> dict[str, Any]:
    """Handle a minimal JSON-RPC MCP request.

    Parameters
    ----------
    payload:
        JSON-RPC request body.

    Returns
    -------
    dict
        JSON-RPC response payload.
    """
    if payload.method == 'initialize':
        result = {
            'protocolVersion': '2025-03-26',
            'serverInfo': {'name': 'roitelet-llm', 'version': '0.1.0'},
            'capabilities': {'tools': {}},
        }
    elif payload.method == 'tools/list':
        result = {
            'tools': [
                {
                    'name': 'roitelet.chat',
                    'description': 'Route a prompt to the best three models, synthesize locally, and return telemetry.',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'prompt': {'type': 'string'},
                            'top_k': {'type': 'integer', 'default': 3},
                            'raw_power': {'type': 'number', 'default': 0.7},
                            'ecofrugality': {'type': 'number', 'default': 0.3},
                            'independence': {'type': 'boolean', 'default': False},
                            'allow_vlms': {'type': 'boolean', 'default': False},
                            'pseudonymize': {'type': 'boolean', 'default': False},
                        },
                        'required': ['prompt'],
                    },
                }
            ]
        }
    elif payload.method == 'tools/call':
        params = payload.params
        if params.get('name') != 'roitelet.chat':
            raise ValueError(f"Unknown tool: {params.get('name')}")
        arguments = params.get('arguments', {})
        response = await run_roitelet_chat(
            ChatRequest(
                prompt=arguments['prompt'],
                top_k=int(arguments.get('top_k', 2)),
                preferences=RouterPreferences(
                    raw_power=float(arguments.get('raw_power', 0.7)),
                    ecofrugality=float(arguments.get('ecofrugality', 0.3)),
                    independence=bool(arguments.get('independence', False)),
                    allow_vlms=bool(arguments.get('allow_vlms', False)),
                    pseudonymize=bool(arguments.get('pseudonymize', False)),
                ),
            )
        )
        result = {
            'content': [
                {
                    'type': 'text',
                    'text': response.synthesis.content,
                }
            ],
            'structuredContent': response.model_dump(),
        }
    else:
        raise ValueError(f'Unsupported MCP method: {payload.method}')

    return {'jsonrpc': '2.0', 'id': payload.id, 'result': result}
