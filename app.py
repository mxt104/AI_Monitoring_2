import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Gemini support
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


DB_PATH = "monitoring_history.db"


# ============================================================
# 1. KNOWLEDGE GRAPH
# ============================================================

class KnowledgeGraph:
    """Small explainable Knowledge Graph using subject-relation-object triples."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.load_default_knowledge()

    def add_fact(self, subject: str, relation: str, obj: str, source: str = "manual"):
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity_or_value")
        self.graph.add_edge(
            subject,
            obj,
            relation=relation,
            source=source
        )

    def load_default_knowledge(self):
        facts = [
            ("Water", "has boiling point", "100°C at standard atmospheric pressure"),
            ("Earth", "has shape", "approximately spherical"),
            ("The Sun", "is", "a star"),
            ("Plants", "need", "sunlight for photosynthesis"),
            ("Electric vehicles", "use", "batteries"),
            ("Diabetes", "requires", "medical management"),
            ("Diabetes", "is not universally cured by", "herbal tea"),
        ]
        for fact in facts:
            self.add_fact(*fact)

    def triples(self) -> List[Tuple[str, str, str]]:
        output = []
        for subject, target, data in self.graph.edges(data=True):
            output.append((subject, data["relation"], target))
        return output

    def facts_as_text(self) -> List[str]:
        return [
            f"{s} {r} {o}."
            for s, r, o in self.triples()
        ]

    def search_entity(self, keyword: str) -> List[Tuple[str, str, str]]:
        keyword = keyword.lower()
        return [
            triple for triple in self.triples()
            if keyword in " ".join(triple).lower()
        ]


# ============================================================
# 2. CONTRADICTION AND SUPPORT RULES
# ============================================================

CONTRADICTION_RULES = [
    {
        "pattern": r"\bwater\s+boils?\s+at\s+(50|200)\s*°?\s*c",
        "evidence": "Water boils at approximately 100°C at standard atmospheric pressure.",
        "reason": "The stated boiling point conflicts with the stored fact."
    },
    {
        "pattern": r"\bearth\s+is\s+flat\b",
        "evidence": "Earth is approximately spherical.",
        "reason": "The statement conflicts with the stored Earth-shape fact."
    },
    {
        "pattern": r"\bsun\s+is\s+a\s+planet\b",
        "evidence": "The Sun is a star.",
        "reason": "The statement assigns an incorrect category to the Sun."
    },
    {
        "pattern": r"\bplants\s+do\s+not\s+need\s+sunlight\b",
        "evidence": "Plants need sunlight for photosynthesis.",
        "reason": "The statement conflicts with the stored photosynthesis fact."
    },
    {
        "pattern": r"\belectric\s+vehicles\s+do\s+not\s+use\s+batteries\b",
        "evidence": "Electric vehicles use batteries.",
        "reason": "The statement conflicts with the stored electric-vehicle fact."
    },
    {
        "pattern": r"\bherbal\s+tea\s+always\s+cures\s+diabetes\b",
        "evidence": "Diabetes requires medical management and is not universally cured by herbal tea.",
        "reason": "The statement makes an unsupported universal medical claim."
    }
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def rule_based_check(statement: str) -> Dict:
    statement = normalize(statement)

    for rule in CONTRADICTION_RULES:
        if re.search(rule["pattern"], statement):
            return {
                "status": "Contradictory",
                "confidence": 0.96,
                "reason": rule["reason"],
                "evidence": rule["evidence"],
                "method": "Rule-based inference",
                "action": "Flag for human review"
            }

    return {
        "status": "Not contradicted by rules",
        "confidence": None,
        "reason": "No predefined contradiction rule was matched.",
        "evidence": "No contradiction rule matched.",
        "method": "Rule-based inference",
        "action": "Continue semantic analysis"
    }


# ============================================================
# 3. SEMANTIC SIMILARITY
# ============================================================

def similarity_check(statement: str, knowledge: KnowledgeGraph) -> Dict:
    facts = knowledge.facts_as_text()

    if not facts:
        return {
            "score": 0.0,
            "closest_fact": "No facts available.",
        }

    documents = [statement] + facts
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    best_index = int(scores.argmax())
    return {
        "score": float(scores[best_index]),
        "closest_fact": facts[best_index]
    }


def final_validation(statement: str, knowledge: KnowledgeGraph) -> Dict:
    rule_result = rule_based_check(statement)

    if rule_result["status"] == "Contradictory":
        return {
            **rule_result,
            "similarity": 0.0
        }

    similarity_result = similarity_check(statement, knowledge)
    score = similarity_result["score"]

    if score >= 0.55:
        status = "Supported / Similar"
        action = "Accept provisionally"
        confidence = min(0.95, score)
    elif score >= 0.25:
        status = "Uncertain"
        action = "Request additional verification"
        confidence = score
    else:
        status = "Unsupported"
        action = "Flag for human review"
        confidence = 1 - score

    return {
        "status": status,
        "confidence": round(float(confidence), 3),
        "reason": (
            "The statement has sufficient textual similarity to stored knowledge."
            if status == "Supported / Similar"
            else "The statement requires additional evidence or review."
        ),
        "evidence": similarity_result["closest_fact"],
        "method": "TF-IDF cosine similarity",
        "action": action,
        "similarity": round(float(score), 3)
    }


# ============================================================
# 4. OPTIONAL GEMINI ANALYSIS
# ============================================================

def gemini_analysis(statement: str) -> Dict:
    """
    Optional AI-assisted analysis.
    Set GEMINI_API_KEY before running the application.
    This output is advisory and must not be treated as guaranteed truth.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not GEMINI_AVAILABLE:
        return {"error": "Install google-genai to enable Gemini analysis."}

    if not api_key:
        return {"error": "GEMINI_API_KEY is not configured."}

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are an AI output monitoring assistant.
Analyze the following statement:

STATEMENT:
{statement}

Return valid JSON with these keys:
classification: one of ["supported", "possibly_unsupported", "possibly_contradictory", "uncertain"]
reason: short explanation
entities: list of important entities
claims: list of claims found in the statement

Do not claim certainty without evidence.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw = response.text.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw).strip()
        return json.loads(raw)

    except Exception as exc:
        return {"error": f"Gemini analysis failed: {exc}"}


# ============================================================
# 5. SQLITE MONITORING LOG
# ============================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                statement TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                method TEXT,
                evidence TEXT,
                action TEXT,
                human_label TEXT,
                feedback TEXT
            )
        """)
        conn.commit()


def save_result(statement: str, result: Dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO monitoring_log
            (timestamp, statement, status, confidence, method, evidence, action)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            statement,
            result.get("status"),
            result.get("confidence"),
            result.get("method"),
            result.get("evidence"),
            result.get("action")
        ))
        conn.commit()


def get_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM monitoring_log ORDER BY id DESC",
            conn
        )


def save_feedback(record_id: int, label: str, feedback: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE monitoring_log
            SET human_label = ?, feedback = ?
            WHERE id = ?
        """, (label, feedback, record_id))
        conn.commit()


# ============================================================
# 6. STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title="AI Output Monitoring System",
    page_icon="🧠",
    layout="wide"
)

init_db()
knowledge_graph = KnowledgeGraph()

st.title("🧠 AI-Driven Output Monitoring System")
st.caption(
    "Educational prototype: Knowledge Graph + rule-based inference + "
    "TF-IDF similarity + optional Gemini analysis"
)

with st.sidebar:
    st.header("System Modules")
    st.write("✅ Knowledge Graph")
    st.write("✅ Rule-Based Inference")
    st.write("✅ Semantic Similarity")
    st.write("✅ Confidence Estimation")
    st.write("✅ SQLite Monitoring Log")
    st.write("✅ Human Feedback")
    st.write("◻ Optional Gemini Analysis")

    st.divider()
    st.subheader("Knowledge Graph Triples")
    triples_df = pd.DataFrame(
        knowledge_graph.triples(),
        columns=["Subject", "Relation", "Object"]
    )
    st.dataframe(triples_df, use_container_width=True, hide_index=True)

tab_validate, tab_history, tab_feedback, tab_ai = st.tabs(
    ["Validate Output", "Monitoring History", "Human Feedback", "Gemini Analysis"]
)

with tab_validate:
    st.subheader("Validate an AI-generated statement")

    examples = [
        "Water boils at 50°C.",
        "Water boils at approximately 100°C at standard atmospheric pressure.",
        "The Sun is a planet.",
        "Electric vehicles use batteries.",
        "Drinking herbal tea always cures diabetes.",
        "Quantum computers are useful."
    ]

    selected = st.selectbox(
        "Choose a demonstration statement",
        ["Custom input"] + examples
    )

    statement = st.text_area(
        "AI-generated output",
        value="" if selected == "Custom input" else selected,
        height=120
    )

    if st.button("🔍 Validate Statement", type="primary"):
        if not statement.strip():
            st.warning("Enter a statement first.")
        else:
            result = final_validation(statement, knowledge_graph)
            save_result(statement, result)

            st.subheader("Validation Result")
            st.metric("Status", result["status"])
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            st.write("**Reason:**", result["reason"])
            st.write("**Evidence:**", result["evidence"])
            st.write("**Method:**", result["method"])
            st.write("**Recommended Action:**", result["action"])

            if result["status"] == "Contradictory":
                st.error("The statement conflicts with a predefined knowledge rule.")
            elif result["status"] == "Unsupported":
                st.warning("The statement could not be sufficiently verified.")
            elif result["status"] == "Uncertain":
                st.info("The statement needs additional verification.")
            else:
                st.success("The statement is provisionally supported.")

            st.info(
                "Important: similarity and rules are indicators, not proof of truth. "
                "Human review is required for high-risk decisions."
            )

with tab_history:
    st.subheader("Monitoring History")
    history = get_history()

    if history.empty:
        st.info("No monitoring records available.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV Log",
            history.to_csv(index=False).encode("utf-8"),
            file_name="monitoring_history.csv",
            mime="text/csv"
        )

with tab_feedback:
    st.subheader("Human-in-the-Loop Feedback")
    history = get_history()

    if history.empty:
        st.info("Validate at least one statement first.")
    else:
        record_map = {
            f"{row['id']} — {row['statement'][:70]}": int(row["id"])
            for _, row in history.iterrows()
        }
        selected_record = st.selectbox("Select a monitoring record", list(record_map))
        record_id = record_map[selected_record]

        human_label = st.selectbox(
            "Human evaluation",
            ["Correct decision", "Incorrect decision", "Needs further review"]
        )
        feedback = st.text_area("Reviewer comment")

        if st.button("Save Human Feedback"):
            save_feedback(record_id, human_label, feedback)
            st.success("Human feedback saved successfully.")

with tab_ai:
    st.subheader("Optional Gemini-Assisted Analysis")
    st.write(
        "This module uses Gemini to extract entities and classify a statement. "
        "It is optional and requires a Gemini API key."
    )

    ai_statement = st.text_area(
        "Statement for Gemini analysis",
        placeholder="Enter a statement to analyze..."
    )

    if st.button("Run Gemini Analysis"):
        if not ai_statement.strip():
            st.warning("Enter a statement first.")
        else:
            with st.spinner("Calling Gemini..."):
                analysis = gemini_analysis(ai_statement)
            st.json(analysis)

st.divider()
st.caption(
    "Limitation: This prototype uses a small manually created Knowledge Graph. "
    "It does not guarantee factual correctness and must not be used for medical, "
    "legal, financial, or safety-critical decisions."
)
