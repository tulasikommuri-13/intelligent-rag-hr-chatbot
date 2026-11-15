# Intelligent RAG Streamlit starter

## Files to add
Copy the HR policy PDFs into `zyro-dynamics-hr-corpus/` in this repository.

Important:
- Do NOT add `01_Employee_Handbook.pdf`; it belongs to a different company (Acrux Dynamics).
- Add the remaining relevant Zyro Dynamics policy PDFs.
- Keep PDF filenames unchanged where possible.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push `app.py`, `requirements.txt`, and `zyro-dynamics-hr-corpus/` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py`.
3. In App settings → Secrets, add:
```toml
GROQ_API_KEY = "your_groq_api_key"
```
4. Deploy and test the public URL.

## Notes
- The app uses PDF page-level text chunks, FAISS, MMR (fetch_k=25, k=5, lambda=0.75), BGE small English embeddings, and Groq.
- Citations show PDF filename, page number, and the retrieved passage. This is not a clickable PDF line-level citation; verify the exact line in the original PDF during the demo.
- The similarity threshold is an initial guardrail and must be tested on in-scope and out-of-scope questions. Do not claim it is fully validated.
- The app skips `01_Employee_Handbook.pdf` because it is not a Zyro Dynamics document.
