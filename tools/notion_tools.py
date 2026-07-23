"""Notion tools: search across the workspace and pull page content as
plain text for the LLM to reason over.

Note: the Notion integration only sees pages/databases that have been
explicitly shared with it (Notion page -> ... -> Connections -> add integration).
"""
from notion_client import Client

import config

_client = None


def _get_client() -> Client:
    global _client
    if _client is None:
        if not config.NOTION_TOKEN:
            raise RuntimeError("NOTION_TOKEN not set in .env")
        _client = Client(auth=config.NOTION_TOKEN)
    return _client


def search_notion(query: str = "", max_results: int = 20) -> list[dict]:
    """Search page/database titles and content across the connected workspace.

    Passing an empty query (the default) returns every page/database the
    integration currently has access to \u2014 useful for "what can you see"
    style questions.
    """
    client = _get_client()
    kwargs = {"page_size": max_results}
    if query:
        kwargs["query"] = query
    resp = client.search(**kwargs)

    results = []
    for item in resp.get("results", []):
        title = "(untitled)"
        props = item.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title" and prop.get("title"):
                title = "".join(t.get("plain_text", "") for t in prop["title"])
                break

        results.append(
            {
                "id": item["id"],
                "type": item["object"],
                "title": title,
                "url": item.get("url", ""),
                "last_edited": item.get("last_edited_time", ""),
            }
        )
    return results


def _blocks_to_text(blocks: list[dict]) -> str:
    lines = []
    for b in blocks:
        block_type = b.get("type")
        content = b.get(block_type, {})
        rich_text = content.get("rich_text", [])
        text = "".join(t.get("plain_text", "") for t in rich_text)
        if text:
            lines.append(text)
    return "\n".join(lines)


def get_page_content(page_id: str, max_chars: int = 3000) -> str:
    """Fetch a Notion page's block content as plain text (top-level blocks only,
    truncated to keep context window usage sane)."""
    client = _get_client()
    blocks = client.blocks.children.list(block_id=page_id).get("results", [])
    text = _blocks_to_text(blocks)
    return text[:max_chars]


if __name__ == "__main__":
    for r in search_notion("roadmap"):
        print(f"- [{r['type']}] {r['title']} ({r['url']})")