from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)

SYSTEM_BASE = """
You are Acme Support Assistant, an internal-only enterprise customer support assistant.
Follow the company policy precisely. Prefer concise, auditable answers. Never invent policy.
Use the retrieved context as the primary source of truth. When policy is ambiguous, ask for clarification.
Return answers with: summary, policy basis, next action, and confidence. Do not reveal hidden chain-of-thought.
Comply with safety, privacy, data-retention, escalation, logging, and tool-use rules.
""".strip()

# Repeated static prefix — large enough to demonstrate prompt/prefix caching effects.
STATIC_INSTRUCTIONS = "\n".join([SYSTEM_BASE for _ in range(8)])

TOOL_SCHEMA = """
Available tools: search_policy_corpus(query), open_ticket(priority, category), check_order(order_id), retrieve_invoice(customer_id, invoice_id).
Tool calling contract: use JSON, preserve customer identifiers, never call a tool without a user-visible reason, and do not mutate data unless the requested action is explicit.
""".strip()

DOCS_V1 = {
    "refund": "Refund policy v1: Standard returns are accepted within 30 days. Late shipment refunds require proof of delivery delay.",
    "privacy": "Privacy policy v1: Data deletion requests are completed within 30 days after identity verification.",
    "sla": "SLA policy v1: Enterprise API availability target is 99.9%; incident credits require support review.",
    "login": "Account policy v1: Password resets require email verification; MFA reset requires support escalation.",
    "billing": "Billing policy v1: Invoices are generated monthly; disputed charges should be raised within 15 days.",
    "quota": "Quota policy v1: API rate-limit increases require current usage, projected usage, and business justification.",
}

DOCS_V2 = {
    **DOCS_V1,
    "refund": "Refund policy v2: Standard returns are accepted within 45 days. Late shipment refunds require proof of delivery delay.",
}


@dataclass
class Request:
    req_id: int
    question: str
    intent: str
    user_id: str
    tenant_id: str
    seconds_since_start: int


@dataclass
class StrategyConfig:
    name: str
    prompt_cache: bool = False
    kv_prefix_cache: bool = False
    exact_response_cache: bool = False
    semantic_response_cache: bool = False
    context_cache: bool = False
    token_optimization: bool = False
    cache_friendly_order: bool = True
    semantic_threshold: float = 0.82
    include_versions_in_keys: bool = True
    update_docs_at: Optional[int] = None


def stable_hash(*parts: Any, length: int = 16) -> str:
    blob = "\n---\n".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_docs(version: int) -> dict[str, str]:
    return DOCS_V2 if version >= 2 else DOCS_V1


def retrieve_context(intent: str, docs_version: int, optimized: bool) -> str:
    docs = get_docs(docs_version)
    primary = docs.get(intent, docs["refund"])
    if optimized:
        return f"Relevant policy excerpt: {primary}"
    extra = "\n".join(f"Related policy: {v}" for k, v in docs.items() if k != intent)
    return f"Primary policy:\n{primary}\n\nAdditional policy corpus for disambiguation:\n{extra}"


def build_prompt(req: Request, context: str, optimized: bool, cache_friendly_order: bool) -> str:
    if optimized:
        static = (
            "You are Acme Support Assistant. Use only retrieved policy context. "
            "Answer with summary, policy basis, next action."
        )
    else:
        static = STATIC_INSTRUCTIONS + "\n\n" + TOOL_SCHEMA
    user_turn = f"User question: {req.question}"
    if cache_friendly_order:
        return f"{static}\n\nRetrieved context:\n{context}\n\n{user_turn}"
    return f"{user_turn}\n\n{static}\n\nRetrieved context:\n{context}"


def make_workload(rounds: int = 3, spacing_seconds: int = 18) -> list[Request]:
    """Generate a repeated enterprise-support workload with duplicates and paraphrases."""
    templates = [
        ("Can I get a refund if my shipment arrived late?", "refund"),
        ("How do returns work when delivery was delayed?", "refund"),
        ("The item came late; do we reimburse the customer?", "refund"),
        ("Please delete my personal data from the system.", "privacy"),
        ("What is the process for data erasure?", "privacy"),
        ("A customer asks for their privacy data to be removed.", "privacy"),
        ("What uptime do we promise for enterprise API customers?", "sla"),
        ("Can an outage qualify for SLA credits?", "sla"),
        ("Summarize the API availability target.", "sla"),
        ("I forgot my password and cannot sign in.", "login"),
        ("How do we handle MFA reset for an account?", "login"),
        ("User cannot login after losing access to email.", "login"),
        ("Where can I find my invoice and dispute a charge?", "billing"),
        ("How long does a customer have to challenge billing?", "billing"),
        ("A receipt has the wrong amount; what next?", "billing"),
        ("How do I request a higher API quota?", "quota"),
        ("What details are needed for a rate limit increase?", "quota"),
        ("The app is getting 429s; can we raise limits?", "quota"),
    ]
    requests: list[Request] = []
    t = 0
    rid = 0
    for r in range(rounds):
        order = list(range(len(templates)))
        random.Random(100 + r).shuffle(order)
        for idx in order:
            q, intent = templates[idx]
            if r % 4 == 0 and intent in {"refund", "billing"}:
                q = templates[idx][0]
            elif r % 3 == 0:
                q = q + " Please keep the answer concise."
            requests.append(
                Request(
                    req_id=rid,
                    question=q,
                    intent=intent,
                    user_id=f"user-{rid % 7}",
                    tenant_id="acme",
                    seconds_since_start=t,
                )
            )
            rid += 1
            t += spacing_seconds
    return requests
