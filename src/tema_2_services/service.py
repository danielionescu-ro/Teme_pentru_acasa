import json
import os
import hashlib
import re
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("USER_AGENT", "MatinalRAGAssistant/1.0")

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
EMBEDDING_N_FEATURES = int(os.environ.get("EMBEDDING_N_FEATURES", "2048"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]
URL_RELEVANCE_KEYWORDS = (
    "matinal",
    "ploiesti",
    "republicii",
    "cafea",
    "coffee",
    "cafenea",
    "cafe",
    "espresso",
    "barista",
    "menu",
    "meniu",
    "pret",
    "price",
    "horeca",
    "prajitor",
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
        self.web_urls = self._filter_relevant_urls(WEB_URLS)

        os.makedirs(DATA_DIR, exist_ok=True)
        # HashingVectorizer are dimensiune fixa si nu necesita fit.
        self.embedder = HashingVectorizer(
            n_features=EMBEDDING_N_FEATURES,
            alternate_sign=False,
            norm=None,
        )

        # Propozitii de referinta multiple pentru a acoperi toate tipurile de intrebari despre cafenea
        self.relevance_refs = self._embed_texts([
            "program cafenea meniu preturi locatie",
            "cafea espresso americano cappuccino pret bautura",
            "cafenea matinal ploiesti informatii servicii",
            "wi-fi masa lucru laptop ambianta cafenea",
            "comenzi bauturi mancare cafenea client",
        ])

        # Prompt de sistem detaliat pentru a ghida raspunsurile LLM-ului
        self.system_prompt = (
            "Tu esti un asistent client al cafenelei MATINAL din Ploiesti, Bd. Republicii. "
            "Raspunzi doar la intrebari despre: meniu si cafea, comenzi, locatia si accesul, programul de functionare, preturi si promotii, ambianta de lucru pentru laptop, reguli si facilități. "
            "Raspunzurile trebuie sa fie prietenoase, scurte si profesionale. "
            "Raspunde EXCLUSIV in limba romana, fara expresii in engleza. "
            "Daca are intrebari de contact direct, suggest clientului sa ne contacteze la telefon sau email. "
            "Nu raspunzi la intrebari irelevante pentru cafenea - sugereaza politicos ca nu sunt in domeniul tau de cunostinte."
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

        if not all_chunks:
            all_chunks = self._load_documents_from_local_index()

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _load_documents_from_local_index(self) -> list[str]:
        """Fallback local: foloseste index.html din proiect daca sursele web nu pot fi parse-ate."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        local_index = os.path.join(project_root, "index.html")
        if not os.path.exists(local_index):
            return []

        try:
            with open(local_index, "r", encoding="utf-8") as f:
                html = f.read()
        except OSError:
            return []

        # Elimina script/style/head pentru a evita zgomotul de CSS/JS, apoi scoate tag-urile HTML.
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<head[^>]*>.*?</head>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        plain_text = re.sub(r"<[^>]+>", " ", html)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()
        return self._chunk_text(plain_text)

    def _is_relevant_url(self, url: str) -> bool:
        """Verifica rapid daca URL-ul pare relevant pentru domeniul cafenea/HoReCa."""
        parsed = urlparse(url)
        haystack = f"{parsed.netloc}{parsed.path}".lower()
        return any(keyword in haystack for keyword in URL_RELEVANCE_KEYWORDS)

    def _filter_relevant_urls(self, urls: list[str]) -> list[str]:
        """Pastreaza doar URL-uri care par relevante; daca nu ramane nimic, foloseste lista initiala."""
        filtered = [url for url in urls if self._is_relevant_url(url)]
        return filtered if filtered else urls

    def _clear_cached_data(self) -> None:
        """Sterge fisierele cache RAG pentru rebuild complet."""
        for path in (CHUNKS_JSON_PATH, FAISS_INDEX_PATH, FAISS_META_PATH):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                continue

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        # Prompt utilizator structurat cu context din RAG
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Context despre Cafenea Matinal:\n{context}\n\n"
                    f"Intrebarea clientului: {user_input}\n\n"
                    f"Raspunde doar in limba romana. "
                    f"Raspunde scurt si util, folosind informatiile din context. "
                    f"Daca informatia nu e in context, spune ca poti contact direct cafenea."
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

            err_text = str(exc)
            if "No endpoints available matching your guardrail restrictions" in err_text:
                return (
                    "Nu pot raspunde acum deoarece setarile de privacy/guardrails din OpenRouter blocheaza endpoint-urile modelului selectat. "
                    "Intra in OpenRouter > Settings > Privacy si permite endpoint-uri compatibile pentru modelul ales, "
                    "sau schimba GROQ_MODEL cu un model disponibil in contul tau."
                )

            return (
                "Nu pot ajunge la modelul de limbaj acum. "
                f"Detaliu tehnic: {err_text}"
            )

    def _fallback_response_from_context(self, user_input: str, context: str) -> str:
        """Fallback simplu cand LLM nu este disponibil: extrage puncte utile din context."""
        if not context.strip():
            return "Nu am gasit informatii suficiente in sursele RAG pentru aceasta intrebare."

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
            return "Nu am gasit informatii suficiente in sursele RAG pentru aceasta intrebare."

        bullets = "\n".join(f"- {ln}" for ln in unique_lines)
        return (
            "Modelul LLM nu este disponibil momentan, dar din sursele incarcate am gasit:\n"
            f"{bullets}\n"
            "Daca vrei, reformulez raspunsul pe scurt pentru intrebarea ta."
        )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings dense cu HashingVectorizer (fara dependinte TensorFlow)."""
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

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea maxima cu propozitiile de referinta despre cafenea Matinal Ploiesti."""
        embedding = self._embed_texts(text.strip())[0]
        return max(self._cosine_similarity(embedding, ref) for ref in self.relevance_refs)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrarea utilizatorului e despre cafenea Matinal si domeniile sale de activitate."""
        return self.calculate_similarity(user_input) >= 0.20

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            return (
                "Bună! Sunt asistentul cafenelei MATINAL. "
                "Pot raspunde la intrebari despre: meniu, preturi, locatie, program, ambianta pentru lucru. "
                "De exemplu: 'Care e programul cafenelei?' sau 'Aveti Wi-Fi?'"
            )

        if not self.is_relevant(user_message):
            return (
                "Scuza-ma! Sunt specializat doar in informatii despre Cafenea MATINAL. "
                "Pot raspunde la intrebari despre meniu, preturi, locatie, program, ambianta. "
                "Intreaba-ma ceva despre cafenea!"
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Asistent RAG Cafenea Matinal")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Porneste modul interactiv de intrebari si raspunsuri.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Sterge cache-ul RAG si reconstruieste datele din WEB_URLS.",
    )
    args = parser.parse_args()

    assistant = RAGAssistant()

    if args.rebuild:
        assistant._clear_cached_data()
        chunks = assistant._load_documents_from_web()
        print(
            "Cache RAG sters. Rebuild finalizat: "
            f"{len(chunks)} chunks din {len(assistant.web_urls)} URL-uri filtrate."
        )

    if args.chat:
        print("=== MOD CHAT INTERACTIV ===")
        print("Scrie intrebarea ta. Pentru iesire, scrie: exit")
        while True:
            user_input = input("Tu: ").strip()
            if user_input.lower() in {"exit", "quit", "iesire"}:
                print("Asistent: La revedere!")
                break
            print(assistant.assistant_response(user_input))
        raise SystemExit(0)

    print("=== TESTE ASISTENT MATINAL ===")
    print("\n[Test 1 - Intrebare relevanta]")
    print(assistant.assistant_response("Care e programul cafenelei Matinal?"))
    
    print("\n[Test 2 - Intrebare relevanta]")
    print(assistant.assistant_response("Aveti Wi-Fi para a lucra cu laptopul?"))
    
    print("\n[Test 3 - Intrebare relevanta]")
    print(assistant.assistant_response("Cat costa o cafea Americano?"))
    
    print("\n[Test 4 - Intrebare IRELEVANTA]")
    print(assistant.assistant_response("Care e capitala Frantei?"))
    
    print("\n[Test 5 - Intrebare IRELEVANTA]")
    print(assistant.assistant_response("Cum se joaca sah?"))
    
    print("\n[Test 6 - Mesaj gol]")
    print(assistant.assistant_response(""))