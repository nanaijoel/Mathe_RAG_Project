import os
import faiss
import pickle
import re
import subprocess
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ==== Setup ====
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4-turbo"

client = OpenAI(api_key=OPENAI_API_KEY)


# ==== Keyword extrahieren (GPT-basiert) ====
def get_main_keyword(text):
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that extracts the most relevant technical keyword "
                    "from a math-related user question. "
                    "Return only a single lowercase keyword, suitable for a filename. "
                    "Avoid generic words like 'bitte', 'frage', 'thema', etc. "
                    "Replace German umlauts and avoid spaces or special characters."
                )},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        keyword = response.choices[0].message.content.strip().lower()
        keyword = keyword.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        keyword = re.sub(r"\W+", "_", keyword)  # alles außer \w wird _
        return keyword[:40]
    except Exception as e:
        print(f"⚠️ GPT-Keyword-Erkennung fehlgeschlagen: {e}")
        return "antwort"


def markdown_to_latex(text):
    lines = text.splitlines()
    latex_lines = []
    in_align = False

    for line in lines:
        line = line.strip()

        # Überschriften in fett
        if re.match(r"^#{1,6}\s", line):
            content = line.lstrip('#').strip()
            latex_lines.append(r"\textbf{" + content + r"}\\")
            continue

        # Bullet Points
        if line.startswith("- "):
            if not in_align:
                latex_lines.append(r"\begin{itemize}")
                in_align = True
            latex_lines.append(r"\item " + line[2:])
            continue

        # Zeilen mit mehreren Gleichungen (mehrere "=" oder "+")
        if line.count("=") >= 1 or line.count("+") >= 3:
            if not in_align:
                latex_lines.append(r"\begin{align*}")
                in_align = True

            # Aufteilen an mehreren Gleichungen (vorsichtig!)
            segments = re.split(r"\s{2,}", line)  # Doppelte Leerzeichen trennen Schritte
            for segment in segments:
                latex_lines.append(segment.strip() + r"\\")
            continue

        # Normale Zeile
        if in_align:
            if r"\item" not in line:
                latex_lines.append(r"\end{align*}")
                in_align = False

        latex_lines.append(line)

    if in_align:
        latex_lines.append(r"\end{align*}")

    result = "\n".join(latex_lines)

    # Falls \item verwendet wurde, in Umgebung einbetten
    if r"\item" in result and r"\begin{itemize}" not in result:
        result = r"\begin{itemize}" + "\n" + result + "\n" + r"\end{itemize}"

    return result



# ==== Embedding erzeugen ====
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=[text.replace("\n", " ")])
    return response.data[0].embedding


# ==== GPT-Antwort ====
def ask_gpt(context, user_question):
    system_prompt = (
        "You are a helpful math tutor. "
        "Use the provided context to answer the question using correct LaTeX formatting. "
        "Do not make up facts – only use the provided content."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
    ]
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ==== Main ====
def main():
    frage = input("Frage eingeben: ").strip()
    if not frage:
        print("Abbruch: Leere Eingabe.")
        return

    keyword = get_main_keyword(frage)
    print(f"Haupt-Schlüsselwort: {keyword}")

    question_vector = np.array(get_embedding(frage)).astype("float32")
    index = faiss.read_index(os.path.join(EMBED_DIR, "faiss.index"))
    with open(os.path.join(EMBED_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    k = 5
    _, indices = index.search(np.array([question_vector]), k)
    context_chunks = [metadata["texts"][i] for i in indices[0]]
    context_text = "\n\n---\n\n".join(context_chunks)

    antwort_roh = ask_gpt(context_text, frage)
    antwort_latex = markdown_to_latex(antwort_roh)

    tex_path = os.path.join(OUTPUT_DIR, f"{keyword}.tex")
    pdf_path = os.path.join(OUTPUT_DIR, f"{keyword}.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage{amsmath,amssymb,amsthm}" + "\n")
        f.write(r"\usepackage[a4paper,margin=2.5cm]{geometry}" + "\n")
        f.write(r"\begin{document}" + "\n\n")
        f.write(r"\section*{" + keyword.replace("_", " ").title() + "}\n\n")
        f.write(antwort_latex + "\n\n")
        f.write(r"\end{document}")

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], cwd=OUTPUT_DIR, check=True)
        if os.path.exists(pdf_path):
            print(f"\nPDF erstellt: {pdf_path}")
        else:
            raise FileNotFoundError("PDF wurde nicht erzeugt.")
    except Exception as e:
        print(f"Fehler bei PDF-Erzeugung: {e}")


if __name__ == "__main__":
    main()
