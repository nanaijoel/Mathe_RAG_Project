import os
import faiss
import pickle
import re
import subprocess
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4-turbo"

client = OpenAI(api_key=OPENAI_API_KEY)

# Keyword extrahieren
def get_main_keyword(text):
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that extracts the most relevant technical keyword "
                    "from a math-related user question. "
                    "Return only a single uppercase keyword, suitable for a filename. "
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
        print(f"GPT-Keyword-Erkennung fehlgeschlagen: {e}")
        return "antwort"

# Embedding erzeugen
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=[text.replace("\n", " ")])
    return response.data[0].embedding

# GPT-Antwort mit LaTeX direkt erzeugen
def ask_gpt(context, user_question):
    system_prompt = (
        "You are a LaTeX expert and math tutor. "
        "Use the CONTEXT to answer the QUESTION. "
        "Format all rules and formulas using proper LaTeX environments. "
        "Use \\begin{itemize} ... \\item ... \\end{itemize} for rule lists, and \\begin{align*} for equations. "
        "Ensure that each rule or equation appears on a separate line, wrapped if necessary using \\resizebox or small."
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

def main():
    frage = input("Frage eingeben: ").strip()
    if not frage:
        print("Abbruch: Leere Eingabe.")
        return

    keyword = get_main_keyword(frage)
    question_vector = np.array(get_embedding(frage)).astype("float32")

    index = faiss.read_index(os.path.join(EMBED_DIR, "faiss.index"))
    with open(os.path.join(EMBED_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    k = 5
    _, indices = index.search(np.array([question_vector]), k)
    context_chunks = [metadata["texts"][i] for i in indices[0]]
    context_text = "\n\n---\n\n".join(context_chunks)

    antwort_latex = ask_gpt(context_text, frage)

    tex_path = os.path.join(OUTPUT_DIR, f"{keyword}.tex")
    pdf_path = os.path.join(OUTPUT_DIR, f"{keyword}.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage{amsmath,amssymb,amsthm}" + "\n")
        f.write(r"\usepackage{mathtools}" + "\n")
        f.write(r"\usepackage{graphicx}" + "\n")
        f.write(r"\usepackage[a4paper,margin=1.5cm]{geometry}" + "\n")
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
