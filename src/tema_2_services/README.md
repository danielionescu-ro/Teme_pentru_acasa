# Tema 2: Asistent RAG - Cafenea Matinal

## Descriere
Acest modul implementeaza un asistent RAG pentru Cafeneaua MATINAL din Ploiesti.
Asistentul filtreaza intrebarile irelevante, extrage context din surse web, face retrieval cu FAISS (sau fallback NumPy) si trimite contextul catre LLM.

Fisier principal: `src/tema_2_services/service.py`

## Versiune Python suportata
- Varianta testata in acest workspace: Python 3.10.11
- Compatibilitate recomandata: Python >= 3.10

Motiv: codul foloseste typing modern (ex: `str | list[str]`) si pachete validate in `.venv` pe 3.10.11.

## Modul de functionare

### 1. Configurare `.env`
Variabile minime:

```env
GROQ_API_KEY=...
GROQ_BASE_URL=https://openrouter.ai/api/v1
GROQ_MODEL=openai/gpt-5.4-nano

DATA_DIR=./data
WEB_URLS=https://224.ro/matinal/;https://exemplu.ro/pagina-cafea
EMBEDDING_N_FEATURES=2048
LLM_MAX_TOKENS=512
```

### 2. Instalare dependinte
Din radacina proiectului:

```bash
pip install -r requirements.txt
```

### 3. Rulare script direct

```bash
python src/tema_2_services/service.py
```

Ruleaza testele built-in din blocul `if __name__ == "__main__":`.

### 4. Mod chat interactiv

```bash
python src/tema_2_services/service.py --chat
```

### 5. Rebuild cache RAG

```bash
python src/tema_2_services/service.py --rebuild
```

## Flux intern (rezumat)
1. Validare configurare API key.
2. Filtrare URL-uri relevante pentru domeniul cafenea/HoReCa.
3. Incarcare documente web; fallback la `index.html` local daca web fail.
4. Chunking text cu `RecursiveCharacterTextSplitter`.
5. Embedding cu `HashingVectorizer`.
6. Retrieval top-k cu FAISS; fallback NumPy daca FAISS nu este disponibil.
7. Prompt catre LLM cu context si reguli de raspuns.
8. Fallback contextual daca LLM nu poate raspunde.

## Diferente fata de fisierul original
Referinta original: https://github.com/dragosbajenaru1001/Teme_pentru_acasa/blob/main/src/tema_2_services/service.py

### Modificari tehnice principale
- Embeddings:
    - Original: TensorFlow + Universal Sentence Encoder (`tensorflow`, `tensorflow_hub`, `USE_MODEL_URL`).
    - Actual: `HashingVectorizer` din scikit-learn, fara dependinte TensorFlow.

- Relevanta:
    - Original: o singura propozitie de referinta (`self.relevance`) si prag fix 0.5.
    - Actual: mai multe propozitii de referinta (`self.relevance_refs`) si prag 0.20.

- Surse RAG:
    - Original: citea direct din `WEB_URLS`.
    - Actual: filtreaza URL-urile cu `URL_RELEVANCE_KEYWORDS` si adauga fallback local din `index.html`.

- Retrieval:
    - Original: se baza pe FAISS importat direct.
    - Actual: import FAISS in runtime cu fallback pe rankare cosine NumPy daca FAISS lipseste.

- Promptare LLM:
    - Original: model hardcodat in apel (`openai/gpt-oss-20b`) si mesaje TODO.
    - Actual: model din env (`GROQ_MODEL`), prompt de sistem complet, mesaje utilizator personalizate.

- Control cost/tokeni:
    - Original: fara limita explicita de tokeni in request.
    - Actual: `LLM_MAX_TOKENS` configurabil din env (default 512), folosit in `chat.completions.create(...)`.

- Tratare erori:
    - Original: mesaj generic la exceptie.
    - Actual: fallback contextual din chunks, plus mesaj explicit pentru erori OpenRouter guardrails/privacy.

- Operare:
    - Original: teste minimale in main.
    - Actual: comenzi CLI `--chat` si `--rebuild` + utilitar `_clear_cached_data`.

## Parametri configurabili importanti
- `DATA_DIR` - locatie cache chunks/index
- `WEB_URLS` - surse pentru RAG
- `EMBEDDING_N_FEATURES` - dimensiune vectorizare hashing
- `GROQ_MODEL` - model LLM folosit
- `LLM_MAX_TOKENS` - limita max tokeni per raspuns

## Observatii
- Daca LLM nu este accesibil (credite insuficiente, guardrails, endpoint blocat), serviciul raspunde cu fallback contextual din datele RAG.
- Pentru raspunsuri live stabile pe OpenRouter, foloseste un model disponibil contului si o limita rezonabila de tokeni.
