
import re

REFUSAL_MESSAGE = (
    "I don't have that information. Please contact ITF directly."
)

# Topics this ITF assistant is designed to support.
ITF_KEYWORDS = {
    "itf",
    "industrial training fund",
    "siwes",
    "industrial training",
    "training programme",
    "training program",
    "training programmes",
    "training programs",
    "training contribution",
    "reimbursement",
    "levy",
    "employer",
    "employers",
    "trainee",
    "trainees",
    "internship",
    "skills development",
    "vocational training",
    "training centre",
    "training center",
    "contribution act",
    "eligibility",
    "mandate",
    "services",
    "mission",
    "vision",
    "core values",
}


def check_input(query: str) -> dict:
    """
    Shield's input guardrail.

    Checks that a question is present and appears relevant
    to ITF before allowing the workflow to continue.
    """

    if not query or not query.strip():
        return {
            "allowed": False,
            "reason": "empty_input",
            "message": "Please enter a question.",
        }

    cleaned_query = query.strip()

    if len(cleaned_query) > 2000:
        return {
            "allowed": False,
            "reason": "input_too_long",
            "message": (
                "Your question is too long. "
                "Please shorten it and try again."
            ),
        }

    normalized_query = cleaned_query.lower()

    # Basic prompt-injection pattern detection.
    suspicious_patterns = [
        r"ignore (all )?(previous|prior|above) instructions",
        r"reveal (your )?(system prompt|hidden instructions)",
        r"show (me )?(your )?(api key|secret key|credentials)",
        r"bypass (the )?(guardrails|safety rules)",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, normalized_query):
            return {
                "allowed": False,
                "reason": "suspicious_instruction",
                "message": (
                    "I can't process that instruction. "
                    "Please ask a question about ITF services "
                    "or training programmes."
                ),
            }

    # Allow questions that explicitly mention ITF-related terms.
    if any(keyword in normalized_query for keyword in ITF_KEYWORDS):
        return {
            "allowed": True,
            "reason": "itf_related",
            "message": "Input passed the initial checks.",
        }

    return {
        "allowed": False,
        "reason": "outside_scope",
        "message": REFUSAL_MESSAGE,
    }


def check_output(answer: str, research: dict) -> dict:
    """
    Shield's output guardrail.

    Prevents an answer from being released when research
    failed or no supporting evidence was retrieved.

    This is a basic check, not a complete factuality verifier.
    """

    if not answer or not answer.strip():
        return {
            "approved": False,
            "answer": REFUSAL_MESSAGE,
            "reason": "empty_answer",
        }

    if research.get("status") != "success":
        return {
            "approved": False,
            "answer": REFUSAL_MESSAGE,
            "reason": "research_not_successful",
        }

    findings = research.get("findings", [])

    if not findings:
        return {
            "approved": False,
            "answer": REFUSAL_MESSAGE,
            "reason": "no_supporting_evidence",
        }

    return {
        "approved": True,
        "answer": answer.strip(),
        "reason": "basic_checks_passed",
    }
