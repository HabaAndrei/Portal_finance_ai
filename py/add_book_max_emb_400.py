import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import numpy as np

client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_or_create_collection(name="carte_investitii_max_emb_400")
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model_emb = SentenceTransformer('thenlper/gte-small')

# client_db.delete_collection(name="carte_investitii_max_emb_400")
# print(client_db.list_collections(), 'sa vedem ce baze de date am')
# quit()

# iau documentul si il citesc pe fiecare pagina
with open('../securities_us_invest.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    nrTotalPagini = len(pdf_reader.pages)

    frazaAdunarePropozitii = ''
    nrAdunatiDeTokeni = 0

    for numarPagina in range(nrTotalPagini):
        pagina = pdf_reader.pages[numarPagina]
        text = pagina.extract_text()
        lungimeaTextului = len(text)
# daca pagina nu are cel putin 300 de caractere sar peste ea
        if lungimeaTextului < 300:
            continue
# impart pagina in propozitii
        listaCuPropozitii = sent_tokenize(text)

        for pozitie in range(len(listaCuPropozitii)):
            propozitie = listaCuPropozitii[pozitie]
            rezultatFunctie = tokenizer.tokenize(propozitie)
            nrTokeni = len(rezultatFunctie)
# daca exista un numar de tockeni mai mic de 400 mai adaug propozitii
            if nrAdunatiDeTokeni < 400:
                nrAdunatiDeTokeni += nrTokeni
                frazaAdunarePropozitii += propozitie
            else:
# fac embedding pe fraza adunata pana in momentul acesta si o adaug in db
                frazaAdunarePropozitii += propozitie
                embeddings = model_emb.encode(frazaAdunarePropozitii)
                ar_interior = np.array([embeddings])
                if len(collection.get()['ids']) < 1:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[frazaAdunarePropozitii],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=['1']
                        )
                    except ZeroDivisionError:
                        print('eroare id 1')
                else:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[frazaAdunarePropozitii],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=[str(len(collection.get()['ids']) + 1)]
                        )
                        print('am adaugat cu succes la cu id', str(len(collection.get()['ids']) + 1))
                    except ZeroDivisionError:
                        print('eroare id mai mare ca 1')

                frazaAdunarePropozitii = ''
                nrAdunatiDeTokeni = 0
