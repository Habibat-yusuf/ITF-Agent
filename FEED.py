
from datetime import datetime, timezone


VALID_RATINGS = {1, 2, 3, 4, 5}


def process_feedback(
    rating: int,
    comment: str = "",
    question: str = "",
) -> dict:
    """
    Feed — Feedback Agent.

    Validates and categorizes user feedback about the
    ITF multi-agent assistant.
    """

    # Validate the rating.
    if isinstance(rating, bool) or not isinstance(rating, int):
        return {
            "agent": "Feed",
            "status": "rejected",
            "message": "Rating must be a whole number from 1 to 5.",
        }

    if rating not in VALID_RATINGS:
        return {
            "agent": "Feed",
            "status": "rejected",
            "message": "Please provide a rating from 1 to 5.",
        }

    # Validate text lengths.
    if not isinstance(comment, str) or not isinstance(question, str):
        return {
            "agent": "Feed",
            "status": "rejected",
            "message": "Feedback and question must be text.",
        }

    comment = comment.strip()
    question = question.strip()

    if len(comment) > 2000 or len(question) > 2000:
        return {
            "agent": "Feed",
            "status": "rejected",
            "message": "Feedback and question must each be 2000 characters or fewer.",
        }

    # Categorize feedback using the rating and comment.
    if rating <= 2:
        category = "needs_improvement"
        priority = "high"
    elif rating == 3:
        category = "mixed"
        priority = "medium"
    else:
        category = "positive"
        priority = "low"

    comment_lower = comment.lower()

    if any(
        word in comment_lower
        for word in ["incorrect", "wrong", "inaccurate", "false"]
    ):
        category = "possible_accuracy_issue"
        priority = "high"

    elif any(
        word in comment_lower
        for word in ["slow", "timeout", "timed out", "loading"]
    ):
        category = "performance_issue"
        priority = "medium"

    elif any(
        word in comment_lower
        for word in ["confusing", "unclear", "hard to understand"]
    ):
        category = "clarity_issue"
        priority = "medium"

    # Return a structured feedback record.
    return {
        "agent": "Feed",
        "status": "success",
        "rating": rating,
        "category": category,
        "priority": priority,
        "comment": comment,
        "question": question,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": "Thank you! Your feedback has been received.",
    }
