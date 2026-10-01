# NEXA — Streaming Live RAG

## Samsung PRISM Hackathon - Theme 4

### PPT + Demo Video

**[View PPT and Demo Video on Google Drive]:  https://drive.google.com/drive/folders/1RYLjAWMb_PPEaUZRZsl6RPCYOLiwi3AF?usp=sharing**

The Google Drive contains:
- Project Presentation (PPT)
- Working Demo Video

---

# About Theme 4 — Streaming Live RAG

The Samsung PRISM Theme 4 problem focuses on building a **Streaming Live RAG** system that can process continuously changing user input instead of waiting for one complete, perfectly structured query.

In a real conversation, users may:

- start speaking before finishing their thought
- add details later
- refine their requirements
- ask multiple things in one request
- continue an earlier conversation

The challenge is therefore not just retrieving information. The system needs to understand **evolving user intent, retrieve relevant evidence, maintain context, and generate grounded answers while the conversation develops.**

---

# Our Solution — NEXA

We converted the Theme 4 problem statement into a working product called **NEXA**.

NEXA treats a conversation as a continuously evolving request instead of processing every message as an independent query.

The core idea is:

> **Refine, Don't Restart.**

For example:

**Turn 1**

> I need to plan a customer workshop in Pune for 30 people.

**Turn 2**

> I also need the cancellation policy and catering options.

Instead of treating Turn 2 as a completely new request, NEXA understands that it belongs to the existing workshop-planning context and refines the answer accordingly.

The same approach is used for multi-turn travel and reimbursement queries.

---

# The NEXA Pipeline

```text
Continuous Transcript
        ↓
T0 Stability Gate
        ↓
T1 Intent & Query Controller
        ↓
Query / Multi-Intent Decomposition
        ↓
      Hybrid Retrieval
       ↙           ↘
     BM25         Dense
       ↘           ↙
        RRF Fusion
            ↓
       Result Merge
            ↓
         Reranking
            ↓
       Session State
            ↓
         Synthesis
            ↓
   Claim-Level Grounding
            ↓
   Verified Answer + Citations
```


# An Interesting Part — Observability Dashboard

NEXA doesn't just show the final answer. It also lets us see what happens inside the RAG pipeline.

The Observability Dashboard provides an interactive view of:
- Intent detection
- Retrieval activity
- BM25 + Dense retrieval
- Session/refinement flow
- Answer versions
- Grounding verification
- Telemetry and execution events

So instead of treating NEXA as a black box, we can inspect how a user query travels through the system and understand why a particular answer was produced.
**The current dashboard is an initial working implementation. A complete, fully interactive observability dashboard with richer visualizations and deeper pipeline controls will be built in the next stage.**


# Setup & Installation

### 1. Environment Setup

⁠ 1. Setup

### Clone the repository
git clone https://github.com/HARSHITHA579-pixel/Streaming-Live-RAG.git
cd Streaming-Live-RAG

### Create a Python 3.11 virtual environment (Strictly use the same Python version and environment)
python3.11 -m venv venv

### Activate it
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt

### Configure environment variables
cp .env.example .env

### Run the Application

Start the FastAPI server with Uvicorn:

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Now you can open the link and start interacting.
