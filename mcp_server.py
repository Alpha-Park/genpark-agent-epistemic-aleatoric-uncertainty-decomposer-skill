"""
MCP Server for Agent Epistemic Aleatoric Uncertainty Decomposer Skill.
"""

import json
import sys
from client import UncertaintyDecomposer

DECOMPOSER = UncertaintyDecomposer()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "decompose_uncertainty",
                    "description": "Decompose ensemble predictions into epistemic and aleatoric uncertainty",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sample_predictions": {
                                "type": "array",
                                "items": {"type": "object"}
                            }
                        },
                        "required": ["sample_predictions"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "decompose_uncertainty":
            res = DECOMPOSER.decompose(args["sample_predictions"])
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
