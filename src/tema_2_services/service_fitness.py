import json
import os
import hashlib
import re
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "FitnessRAGAssistant/1.0")

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

DATA_DIR = os.environ.get("FITNESS_DATA_DIR", "./data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "fitness_data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "fitness_faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "fitness_faiss.index.meta")
EMBEDDING_N_FEATURES = int(os.environ.get("FITNESS_EMBEDDING_N_FEATURES", "2048"))
LLM_MAX_TOKENS = int(os.environ.get("FITNESS_LLM_MAX_TOKENS", "512"))

DEFAULT_FITNESS_WEB_URLS = [
    "https://unefs.ro",
    "https://umfcd.ro",
    "https://umfcluj.ro",
    "https://www.umfiasi.ro",
    "https://www.reginamaria.ro/articole-medicale/ce-este-miscarea-fizica-si-de-ce-este-importanta",
    "https://www.medlife.ro/articole-medicale/beneficiile-sportului",
    "https://ro.wikipedia.org/wiki/Exerci%C8%9Biu_fizic",
    "https://ro.wikipedia.org/wiki/Fitness",
    "https://ro.wikipedia.org/wiki/Antrenament_cu_greut%C4%83%C8%9Bi",
    "https://ro.wikipedia.org/wiki/Cardio",
    "https://ro.wikipedia.org/wiki/Protein%C4%83",
]

FITNESS_WEB_URLS = [u for u in os.environ.get("FITNESS_WEB_URLS", "").split(";") if u]

URL_RELEVANCE_KEYWORDS = (
    "fitness",
    "antrenament",
    "exercitii",
    "exercitii",
    "forta",
    "cardio",
    "hiit",
    "stretching",
    "mobilitate",
    "postura",
    "kinetoterapie",
    "recuperare",
    "nutritie",
    "proteine",
    "calorii",
    "masa-musculara",
    "slabit",
    "deficit",
    "unefs",
    "iefs",
    "umf",
    "medicina",
    "sport",
)

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL"))
        self.groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b:free")

        seed_urls = FITNESS_WEB_URLS or DEFAULT_FITNESS_WEB_URLS
        self.web_urls = self._filter_relevant_urls(seed_urls)

        os.makedirs(DATA_DIR, exist_ok=True)
        # HashingVectorizer are dimensiune fixa si nu necesita fit.
        self.embedder = HashingVectorizer(
            n_features=EMBEDDING_N_FEATURES,
            alternate_sign=False,
            norm=None,
        )

        self.relevance_refs = self._embed_texts([
            "antrenament fitness pentru slabit cu exercitii cardio si forta",
            "plan de crestere masa musculara cu progresie si aport proteic",
            "program pentru incepatori cu tehnica corecta si recuperare",
            "recomandari despre mobilitate, incalzire si prevenirea accidentarilor",
        ])

        self.system_prompt = (
            "Esti un asistent educational de fitness si nutritie sportiva. "
            "Raspunzi in limba romana, clar si practic, pentru persoane incepatoare, intermediare sau avansate. "
            "Poti discuta despre exercitii, planuri de antrenament, recuperare, nutritie de baza, progresie si prevenirea accidentarilor. "
            "Nu dai diagnostic medical si nu inlocuiesti consultul unui medic. "
            "Daca apar simptome severe, dureri persistente sau afectiuni, recomanda consult medical/fizioterapie."
        )


    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in self.web_urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    all_chunks.extend(chunks)
            except Exception:
                continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _is_relevant_url(self, url: str) -> bool:
        parsed = urlparse(url)
        haystack = f"{parsed.netloc}{parsed.path}".lower()
        return any(keyword in haystack for keyword in URL_RELEVANCE_KEYWORDS)

    def _filter_relevant_urls(self, urls: list[str]) -> list[str]:
        filtered = [url for url in urls if self._is_relevant_url(url)]
        return filtered if filtered else urls

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    "Context fitness (extras din surse web in romana):\n"
                    f"{context}\n\n"
                    f"Intrebarea utilizatorului: {user_input}\n\n"
                    "Raspunde structurat cu: obiectiv, recomandari concrete, exemplu de plan (seturi/repetari/frecventa), "
                    "sfaturi de siguranta si semnale cand e util consult medical."
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.groq_model,
                max_tokens=LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as exc:
            fallback = self._fallback_response_from_context(user_input, context)
            if fallback:
                return fallback
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                f"Te rog incearca din nou in cateva momente. Detaliu tehnic: {exc}"
            )

    def _fallback_response_from_context(self, user_input: str, context: str) -> str:
        """Fallback simplu cand LLM nu este disponibil: extrage puncte utile din context."""
        if not context.strip():
            return "Nu am gasit informatii suficiente in sursele fitness pentru aceasta intrebare."

        query_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9ăâîșț]+", user_input)
            if len(token) >= 3
        }
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

        scored = []
        for sentence in sentences:
            s_lower = sentence.lower()
            overlap = sum(1 for token in query_tokens if token in s_lower)
            if overlap > 0:
                scored.append((overlap, sentence))

        if not scored:
            scored = [(0, s) for s in sentences]

        unique_lines = []
        seen = set()
        for _, sentence in sorted(scored, key=lambda x: x[0], reverse=True):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < 20:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_lines.append(cleaned[:180])
            if len(unique_lines) >= 4:
                break

        if not unique_lines:
            return "Nu am gasit informatii suficiente in sursele fitness pentru aceasta intrebare."

        bullets = "\n".join(f"- {ln}" for ln in unique_lines)
        return (
            "Modelul LLM nu este disponibil momentan, dar din sursele fitness am gasit:\n"
            f"{bullets}\n"
            "Daca vrei, reformulez un plan simplu pe baza acestor puncte."
        )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings dense cu HashingVectorizer."""
        if isinstance(texts, str):
            texts = [texts]
        sparse_vectors = self.embedder.transform(texts)
        return sparse_vectors.astype("float32").toarray()

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return chunks if chunks else [""]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]):
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        import faiss

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": f"hashing_vectorizer_{EMBEDDING_N_FEATURES}",
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        try:
            import faiss
        except Exception:
            return self._retrieve_relevant_chunks_fallback(chunks, user_query, k)

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def _retrieve_relevant_chunks_fallback(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        """Fallback fara FAISS: rankare cosine direct in NumPy."""
        if not chunks:
            return []

        query_embedding = self._embed_texts(user_query).astype("float32")
        chunk_embeddings = self._embed_texts(chunks).astype("float32")

        q = query_embedding[0]
        q_norm = np.linalg.norm(q)
        c_norms = np.linalg.norm(chunk_embeddings, axis=1)
        denom = c_norms * q_norm
        denom[denom == 0] = 1e-12
        sims = np.dot(chunk_embeddings, q) / denom

        k = min(k, len(chunks))
        if k == 0:
            return []

        top_idx = np.argsort(-sims)[:k]
        return [chunks[i] for i in top_idx]

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea maxima fata de referinte fitness."""
        embedding = self._embed_texts(text.strip())[0]
        return max(self._cosine_similarity(embedding, ref) for ref in self.relevance_refs)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrarea utilizatorului este despre fitness/nutritie sportiva."""
        return self.calculate_similarity(user_input) >= 0.20

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            return (
                "Salut! Pot sa te ajut cu exercitii, planuri de antrenament, slabit, crestere masa musculara, "
                "mobilitate si recuperare. De exemplu: 'Ce antrenament full body fac de 3 ori pe saptamana?'"
            )

        if not self.is_relevant(user_message):
            return (
                "Pot raspunde doar la intrebari despre fitness, antrenament si nutritie sportiva. "
                "Incearca o intrebare precum: 'Cate proteine sa consum daca vreau masa musculara?'"
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    print("=== TESTE ASISTENT FITNESS ===")
    print("\n[Test 1 - Intrebare relevanta]")
    print(assistant.assistant_response("Ce exercitii recomanzi pentru slabit la incepatori?"))

    print("\n[Test 2 - Intrebare relevanta]")
    print(assistant.assistant_response("Cate proteine sa consum pe zi pentru masa musculara?"))

    print("\n[Test 3 - Intrebare irelevanta]")
    print(assistant.assistant_response("Care e capitala Frantei?"))