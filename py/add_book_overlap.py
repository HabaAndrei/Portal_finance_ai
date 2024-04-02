import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import CharacterTextSplitter

client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_or_create_collection(name="carte_investitii_overlap")
model_emb = SentenceTransformer('thenlper/gte-small')


text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

with open('../mutual_fonds_and_etfs.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    nrTotalPagini = len(pdf_reader.pages)


    for numarPagina in range(nrTotalPagini):
        pagina = pdf_reader.pages[numarPagina]
        text_din_pag = pagina.extract_text()
        lungimeaTextului = len(text_din_pag)
        if lungimeaTextului < 300:
            continue

        text_cu_split = text_splitter.create_documents([text_din_pag])
        for propozitie in text_cu_split:
            # print('======>>>>>>>', propozitie.page_content , '<<<<=========')
            embeddings = model_emb.encode(propozitie.page_content)
            ar_interior = np.array([embeddings])
            if len(collection.get()['ids']) < 1:
                try:
                    collection.add(
                        embeddings=np.array(ar_interior),
                        documents=[propozitie.page_content],
                        metadatas=[{"source": "mutual_fonds_and_etfs"}],
                        ids=['1']
                    )
                except ZeroDivisionError:
                    print('eroare id 1')
            else:
                try:
                    collection.add(
                        embeddings=np.array(ar_interior),
                        documents=[propozitie.page_content],
                        metadatas=[{"source": "mutual_fonds_and_etfs"}],
                        ids=[str(len(collection.get()['ids']) + 1)]
                    )
                    print('am adaugat cu succes cu id', str(len(collection.get()['ids']) + 1))
                except ZeroDivisionError:
                    print('eroare id mai mare ca 1')

