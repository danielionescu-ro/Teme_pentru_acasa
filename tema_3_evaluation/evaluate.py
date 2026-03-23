from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from tema_3_evaluation.groq_llm import GroqDeepEval
from tema_3_evaluation.report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    # quick test case
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input="Care sunt cele mai bune exercitii pentru a slabi rapid?"
    ),
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input="Cate proteine ar trebui sa consum zilnic daca vreau "
              "sa construiesc masa musculara?"
    ),
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input="Ce program de antrenament recomanzi pentru un incepator "
              "care vrea să fie mai activ?"
    ),
    # Persona test cases
    # Persoana 1 - Incepator, vrea sa slabeasca rapid, crede ca e usor
    LLMTestCase(
        input="Buna! Am auzit de fitness si vreau sa slabesc repede. "
              "Cat de greu e? Daca merg la sala de 2-3 ori pe saptamana "
              "scap de 10 kg intr-o luna?"
    ),
    LLMTestCase(
        input="Salut! Un prieten mi-a zis ca daca fac cateva exercitii "
              "la sala slabesc foarte rapid. Ce exercitii simple sa fac "
              "ca sa slabesc 5 kg cat mai repede posibil?"
    ),
    LLMTestCase(
        input="Vreau sa incep fitness-ul saptamana viitoare. "
              "Stiu ca e simplu, doar ridici greutati si alergi putin. "
              "In cat timp pot sa slabesc vizibil fara sa ma chinui prea mult?"
    ),

    # Persoana 2 - A tinut regim si exercitii fara program, vrea sa slabeasca
    LLMTestCase(
        input="Am tinut dieta cateva luni si am facut si exercitii "
              "dar fara un program clar. Uneori saream peste mese, "
              "alteori mancam foarte putin. Nu am slabit cat am vrut. "
              "Ce gresesc si cum sa procedez corect?"
    ),
    LLMTestCase(
        input="Am combinat un regim strict cu alergat si exercitii acasa "
              "dar fara sa urmez un plan anume. Am slabit putin la inceput "
              "apoi m-am blocat. De ce nu mai slabesc si ce ar trebui sa schimb?"
    ),
    LLMTestCase(
        input="Fac exercitii in fiecare zi, uneori cardio, uneori greutati, "
              "si mananc putine calorii dar rezultatele sunt slabe. "
              "Nu am urmat niciodata un program structurat. "
              "Cum imi construiesc un plan corect de slabit?"
    ),

    # Persoana 3 - Sportiva, cunoaste fitness si proteine, vrea program complex
    LLMTestCase(
        input="Sunt sportiv activ cu experienta in sala, cunosc macronutrientii "
              "si rolul proteinelor in sinteza musculara si in deficit caloric. "
              "Vreau un program complex de cutting de 12 saptamani cu periodizare, "
              "urmat de o faza de stabilizare de 4 saptamani. Poti detalia?"
    ),
    LLMTestCase(
        input="Am experienta cu antrenamentele de forta si stiu cum functioneaza "
              "proteinele in curele de slabire: pastreaza masa musculara in deficit. "
              "Vreau un program de slabit cu split de antrenament, "
              "target caloric si proteic zilnic si o strategie de reverse dieting "
              "pentru stabilizare dupa cutting."
    ),
    LLMTestCase(
        input="Ca sportiv avansat cu cunostinte solide de nutritie si fitness, "
              "caut un program structurat care sa combine antrenament de forta "
              "cu cardio HIIT pentru maximizarea arderii de grasime, "
              "cu mentinerea masei musculare prin aport proteic optim, "
              "si o faza de stabilizare progresiva dupa atingerea greutatii tinta."
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    # Answer Relevancy
    name="Answer Relevancy",
    criteria="""
    Evalueaza daca raspunsul generat este direct si complet relevant
    pentru intrebarea utilizatorului despre fitness sau nutritie.
    Un raspuns bun trebuie sa:
    - Raspunda explicit la ce a intrebat utilizatorul
    - Nu contina informatii irelevante sau deviatii de la subiect
    - Fie proportional ca lungime si detaliu cu complexitatea intrebarii
    - Adreseze toate aspectele mentionate in intrebare    
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    # Helpfulness
    name="Helpfulness",
    criteria=""" 
    Evalueaza daca raspunsul este cu adevarat util si practic pentru
    utilizatorul care cauta sfaturi de fitness sau nutritie.
    Un raspuns util trebuie sa:
    - Ofere sfaturi concrete si aplicabile, nu doar informatii generale
    - Fie adaptat la nivelul si contextul utilizatorului
    - Includa pasi sau recomandari clare pe care utilizatorul le poate urma
    - Ofere valoare reala, nu raspunsuri vagi sau evazive   
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator3 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    # Faithfulness
    name="Faithfulness",
    criteria=""" 
    Evalueaza daca raspunsul contine informatii corecte din punct de
    vedere stiintific si medical legat de fitness si nutritie.
    Un raspuns faithful trebuie sa:
    - Fie bazat pe principii reale de nutritie si exercitiu fizic
    - Nu contina afirmatii false sau exagerate
    - Nu inventeze date, studii sau statistici care nu exista
    - Fie consistent cu ghidurile acceptate de sanatate si fitness
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator4 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    # Bias
    name="Bias",
    criteria="""
    Evalueaza daca raspunsul este lipsit de bias, discriminare
    sau stereotipuri legate de fitness si corp.
    Un raspuns fara bias trebuie sa:
    - Nu favorizeze un anumit gen, varsta sau tip de corp
    - Foloseasca un limbaj incluziv si neutru
    - Nu contina stereotipuri despre aspect fizic sau capacitate atletica
    - Nu judece sau stigmatizeze persoanele supraponderale sau incepatorii
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator5 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    # Correctness
    name="Correctness",
    criteria="""
    Evalueaza daca raspunsul este corect din punct de vedere tehnic
    si foloseste terminologia corecta de fitness si nutritie.
    Un raspuns corect trebuie sa:
    - Foloseasca termenii tehnici corect (cutting, periodizare, macronutrienti)
    - Ofere valori numerice precise si realiste
    - Descrie corect mecanismele fiziologice implicate
    - Fie consistent intern fara contradictii
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()
        if data.get("detail") != "Raspunsul de chat a expirat":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data


async def _run_evaluation() -> tuple[list[dict], list[float], list[float], list[float], list[float], list[float]]:
    results: list[dict] = []
    scores1: list[float] = []
    scores2: list[float] = []
    scores3: list[float] = []
    scores4: list[float] = []
    scores5: list[float] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)
            case.actual_output = (
                candidate.get("response", str(candidate))
                if isinstance(candidate, dict)
                else str(candidate)
            )

            evaluator1.measure(case)
            evaluator2.measure(case)
            evaluator3.measure(case)
            evaluator4.measure(case)
            evaluator5.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            print(f"  Answer Relevancy : {evaluator1.score:.2f}")
            print(f"  Helpfulness      : {evaluator2.score:.2f}")
            print(f"  Faithfulness     : {evaluator3.score:.2f}")
            print(f"  Bias             : {evaluator4.score:.2f}")
            print(f"  Correctness      : {evaluator5.score:.2f}")
            
            results.append({
                "input": case.input,
                "response": case.actual_output,
                "relevanta_score":     evaluator1.score,
                "relevanta_reason":    evaluator1.reason,
                "helpfulness_score":   evaluator2.score,
                "helpfulness_reason":  evaluator2.reason,
                "faithfulness_score":  evaluator3.score,
                "faithfulness_reason": evaluator3.reason,
                "bias_score":          evaluator4.score,
                "bias_reason":         evaluator4.reason,
                "correctness_score":   evaluator5.score,
                "correctness_reason":  evaluator5.reason,
            })
            
            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)
            scores3.append(evaluator3.score)
            scores4.append(evaluator4.score)
            scores5.append(evaluator5.score)

    return results, scores1, scores2, scores3, scores4, scores5


def run_evaluation() -> None:
    results, scores1, scores2, scores3, scores4, scores5 = asyncio.run(_run_evaluation())
    output_file = save_report(results, scores1, scores4, THRESHOLD)
    print(f"\nRaport salvat in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
