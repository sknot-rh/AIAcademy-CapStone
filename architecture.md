# System Architecture: Agentic RAG Chatbot

## 🏗️ High-Level Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system. 
Unlike a linear RAG pipeline that blindly answers questions from retrieved context, this system acts as an autonomous agent. 
It employs **self-reflection**, **dynamic routing**, and **error correction** to ensure high-quality responses.

The system is built on a **pure Python orchestration layer** (replacing rigid graph frameworks for clarity) and leverages local LLMs for privacy and cost control.

---

## 🧩 System Diagram (Logic Flow)

```text
       [User Input]
            |
            v
     +--------------+
     | ORCHESTRATOR |
     +--------------+
            |
            v
     +--------------+
     | 1. RETRIEVE  | <---- (Interacts with Qdrant Vector DB)
     +------+-------+
            |
            v
     +------+-------+
     | 2. GRADER    | (LLM Checks Relevance)
     +------+-------+
            |
      +-----+------------------------+
      |                              |
 (Relevant: YES)               (Relevant: NO)
      |                              |
      v                              v
 +---------+                 +--------------+
 | 3. RAG  |                 | 4. WEB SEARCH| <-----------+
 |   GEN   |                 | (DuckDuckGo) |             |
 +----+----+                 +-------+------+             |
      |                              |                    |
      |                              |                    |
      v                              v                    |
      |                              |                    |
      +-------------+----------------+                    |
                    |                                     |
                    v                                     |
             +------+------+                              |
             | 5. EVALUATOR|                              |
             | (Score 1-5) |                              |
             +------+------+                              |
                    |                                     |
            +-------+-------+                             |
            | Is Score > 3? |                             |
            +-------+-------+                             |
                    |                                     |
          (YES)     |     (NO - If RAG Failed)            |
            |       +-------------------------------------+
            v                    (Retry Loop)
             +------+------+                              
             | 6. Save file|                              
             +------+------+                              
                    |                                     
                    v                                     
           +--------+--------+                          
           | 7. Generate post|                          
           +--------+--------+  
                    |
                    v 
            (( FINAL OUTPUT ))