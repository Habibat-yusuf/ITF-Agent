
from RAG import retrieve_context


def research_itf(
    query: str,
    match_count: int = 5,
) -> dict:
    """
    Rese — Researcher Agent.

    Searches the ITF knowledge base and returns relevant evidence
    for Wry, the Writer Agent.
    """

    if not query or not query.strip():
        return {
            "agent": "Rese",
            "status": "rejected",
            "query": query,
            "findings": [],
            "sources": [],
            "message": "A valid research question is required.",
        }

    try:
        result = retrieve_context(
            query=query.strip(),
            match_count=match_count,
            match_threshold=0.3,
        )

        chunks = result.get("chunks", [])

        findings = [
            {
                "content": item.get("content", ""),
                "source": item.get("source", "Unknown source"),
            }
            for item in chunks
            if item.get("content", "").strip()
        ]

        if not findings:
            return {
                "agent": "Rese",
                "status": "no_evidence",
                "query": query,
                "findings": [],
                "sources": [],
                "message": (
                    "No sufficiently relevant information was found "
                    "in the ITF knowledge base."
                ),
            }

        return {
            "agent": "Rese",
            "status": "success",
            "query": query,
            "findings": findings,
            "sources": result.get("sources", []),
            "message": (
                f"Retrieved {len(findings)} relevant evidence passages."
            ),
        }

    except Exception:
        # Avoid exposing credentials or internal infrastructure details.
        return {
            "agent": "Rese",
            "status": "error",
            "query": query,
            "findings": [],
            "sources": [],
            "message": (
                "Research could not be completed. Please try again."
            ),
        }
