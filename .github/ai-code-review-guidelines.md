# Code Review Guidelines

⚠️ **Important**  
This document defines expectations for AI-assisted or human-led code reviews.
It is **not** a GitHub Copilot instruction file and must not be treated as one.

---

## Purpose

This document describes the standard approach for performing a **production‑grade review**
of Python applications using Large Language Models (LLMs) or human reviewers.

The goal is to ensure code is:

- Functionally correct
- Secure and reliable
- Performant and maintainable
- Production-ready

---

## Three‑Pass Review Method

All full reviews should be performed in **three explicit passes**, followed by a consolidated report.

---

### Pass 1 — Correctness & Architecture

Focus on:

- Functional correctness
  - Logic errors, incorrect assumptions, race conditions
  - Edge cases (empty inputs, None, large data, ordering, encoding, time/date)
- Architecture & structure
  - Separation of concerns, layering, coupling/cohesion
  - Module boundaries and dependency direction
- API / interface correctness
  - Function contracts, return types, error semantics
- Concurrency / async correctness (if applicable)
  - Blocking I/O, thread safety, async/await misuse

**Expected output:**

- Findings grouped by file/module/function
- Severity: Critical / High / Medium / Low
- Impact and concrete fix suggestions
- Code snippets or diff-style suggestions for key issues

---

### Pass 2 — Security & Reliability

Focus on:

- Security risks
  - Injection vectors, unsafe deserialization, path traversal, SSRF
  - Auth/authz gaps, secrets handling, sensitive logging
- Threat modeling
  - How untrusted data enters, flows through, and exits the system
- Reliability
  - Error handling, retries/backoff, timeouts, idempotency
  - Resource cleanup and failure modes
- Observability basics
  - Structured logging, correlation IDs, safe exception reporting

**Expected output:**

- Prioritized list of security and reliability issues
- Clear remediation steps
- Identification of missing controls (timeouts, validation, secrets management)

---

### Pass 3 — Performance & Maintainability

Focus on:

- Performance
  - Algorithmic complexity, unnecessary I/O, N+1 patterns
  - Caching, batching, streaming vs loading
- Concurrency & throughput
  - Blocking calls, async performance, connection pooling
- Memory usage
  - Large objects, copies, lifecycle management
- Maintainability
  - Readability, naming, typing, docstrings
  - Modularity, testability, configuration hygiene

**Expected output:**

- Suggested profiling targets and metrics
- Targeted refactors with example code
- Tooling recommendations (linting, typing, testing)

---

## Final Report Requirements

Each full review must conclude with:

1. **Executive Summary**
   - 5–10 bullets covering readiness and top risks

2. **Risk Register**
   - Ranked issues with severity, impact, and remediation

3. **Quick Wins**
   - Small changes with high impact

4. **Medium‑Term Refactors**
   - Structural or architectural improvements

5. **Test Plan**
   - Unit, integration, and edge-case coverage

6. **Operational Notes (if relevant)**
   - Config, secrets, logging, monitoring, CI checks

---

## Review Principles

- Be explicit about assumptions when runtime context is missing
- Prefer actionable, code-level feedback over generic advice
- Use minimal, incremental changes where possible
- Treat AI output as advisory, not authoritative

---
