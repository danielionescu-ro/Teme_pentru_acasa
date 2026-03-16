# Tema 2: Asistent RAG - Cafenea Matinal Ploiești

## 📋 Descriere
Asistent inteligent cu Retrieval-Augmented Generation (RAG) pentru cafenea **MATINAL** situată pe **Bd. Republicii, Ploiești**. Asistentul răspunde **doar la întrebări relevante** pentru domeniul de activitate al cafenelei.

## 🎯 Funcionalități Implementate

### 1. **Relevance Detection** (Detecția Relevanței)
- ✅ Propozitie de referință specifică: *"Aceasta este o intrebare despre cafeneaua Matinal din Ploiesti - meniu, cafea, comenzi, locatie, program, preturi, ambianta de lucru, produse si servicii."*
- ✅ Threshold de similaritate: **0.45** (echilibru între acceptare si filtrare)
- ✅ Embeddings: Universal Sentence Encoder

### 2. **System Prompt** (Instrucții LLM)
Asistentul este ghidat să răspundă la:
- Meniu și tipuri de cafea
- Comenzi și procesul de vânzare
- Locație și acces
- Program de funcționare
- Prețuri și promoții
- Brigadă de lucru (Wi-Fi, mese, prize, ambianță)
- Reguli și facilități

Respinge **politicos** întrebări irelevante.

### 3. **User Prompt** (Template Mesaj Utilizator)
Structura mesajului includeL
- Context din RAG (chunks relevante)
- Întrebarea clientului
- Indicații pentru răspuns scurt și util

### 4. **Error Messages** (Mesaje Ghidare)
- ✅ Mesaj pentru input gol
- ✅ Mesaj pentru întrebări irelevante
- ✅ Mesaj pentru erori de conectare la LLM

### 5. **Test Queries** (Teste Comprehensive)
```
✓ Test 1: "Care e programul cafenelei Matinal?" (RELEVANT)
✓ Test 2: "Aveti Wi-Fi para a lucra cu laptopul?" (RELEVANT)
✓ Test 3: "Cat costa o cafea Americano?" (RELEVANT)
✗ Test 4: "Care e capitala Frantei?" (IRRELEVANT)
✗ Test 5: "Cum se joaca sah?" (IRRELEVANT)
✓ Test 6: "" (EMPTY - teste mesaj gol)
```

## 🔧 Setup & Configurare

### 1. Instalare Dependențe
```bash
pip install -r requirements.txt
```

### 2. Configurare Environment
Creează fișierul `.env` (sau copy din `.env.example`):

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Data Directory
DATA_DIR=/app/data

# URLs pentru RAG (separate prin ;)
# IMPORTANT: Adaugă link-urile cu informații despre Cafenea Matinal
WEB_URLS=https://example.com/matinal;https://facebook.com/cafenea-matinal

# Model URL
USE_MODEL_URL=https://tfhub.dev/google/universal-sentence-encoder/4
```

### 3. Adaugă Link-urile Cafenelei
Modifică `WEB_URLS` cu link-urile reale:
- Site propriu al cafenelei (dacă există)
- Pagina Facebook / Instagram
- Google Business Profile
- Platforme de review (TripAdvisor, Google Maps)
- Blog cu detalii despre meniu, program, preturi

## 🚀 Rulare & Testare

### Rulare teste built-in:
```bash
python service.py
```

Output așteptat:
```
=== TESTE ASISTENT MATINAL ===

[Test 1 - Intrebare relevanta]
Bună! Programul cafenelei Matinal este...

[Test 2 - Intrebare relevanta]
Da, avem Wi-Fi gratuit pentru clienți...

[Test 3 - Intrebare relevanta]
Americano costă 12 RON...

[Test 4 - Intrebare IRELEVANTA]
Scuza-ma! Sunt specializat doar in informatii despre Cafenea MATINAL...

[Test 5 - Intrebare IRELEVANTA]
Scuza-ma! Sunt specializat doar in informatii despre Cafenea MATINAL...

[Test 6 - Mesaj gol]
Bună! Sunt asistentul cafenelei MATINAL...
```

## 📊 Arhitectura RAG

```
User Input
    ↓
[Relevance Check] ← Compare cu referință embedding
    ↓
  Relevant? → No → Mesaj "Scuza-ma!"
    ↓ Yes
[Load Web Docs] → WebBaseLoader din WEB_URLS
    ↓
[Chunk Text] → RecursiveCharacterTextSplitter (300 chars)
    ↓
[Build/Load FAISS Index] → Embeddings cache
    ↓
[Retrieve Top-K] → Similaritate cosine cu query
    ↓
[Send to LLM] → Context + System Prompt + User Query
    ↓
Response
```

## 🎛️ Parametri Configurabili

| Parametru | Valoare | Descriere |
|-----------|---------|-----------|
| `chunk_size` | 300 | Dimensiune chunk text |
| `chunk_overlap` | 20 | Suprapunere între chunks |
| `k` (retrieval) | 5 | Număr chunks relevante |
| `similarity_threshold` | 0.45 | Prag relevație (0-1) |
| `model` | openai/gpt-oss-20b | Model LLM via Groq |

## 📝 Observații Importante

1. **Caching**: Index FAISS este cache-uit pe disc pentru performanță
2. **Hash validation**: Verifycare că URL-urile nu s-au schimbat
3. **Error Handling**: Excepții la LLM sunt captate și raportate corespunzător
4. **Romanian Support**: Cod și mesaje 100% în limba română

## ✅ Cerințe Tema Completate

- [x] Propozitie referință specifică pentru domeniu
- [x] System prompt detaliat
- [x] User prompt cu context RAG
- [x] Mesaje eroare customizate
- [x] Query-uri test (relevante + irelevante)
- [x] URL-uri web pentru RAG
- [x] Filtrare relevanță

---

**Autor**: Elev - Tema 2  
**Data**: Martie 2026  
**Domeniu**: Cafenea MATINAL, Ploiești, Bd. Republicii
