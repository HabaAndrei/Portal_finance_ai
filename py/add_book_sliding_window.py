import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import numpy as np

client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_or_create_collection(name="carte_investitii_sliding_window")
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model_emb = SentenceTransformer('thenlper/gte-small')

with open('../securities_us_invest.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    nrTotalPagini = len(pdf_reader.pages)

    tokeni_100 = 0
    fraza_100 = ''
    tokeni_400 = 0
    fraza_400 = ''
    for numarPagina in range(nrTotalPagini):
        pagina = pdf_reader.pages[numarPagina]
        text = pagina.extract_text()
        lungimeaTextului = len(text)
        listaCuPropozitii = sent_tokenize(text)

        if lungimeaTextului < 400:
            continue


        for prop in listaCuPropozitii:
            propozitie = ''.join(prop.split('\n'))
            nrTokeni = len(tokenizer.tokenize(propozitie))

################# pentru fraza cu 100 de tokeni
            if (tokeni_100 < 80):

                tokeni_100 += nrTokeni
                fraza_100 += propozitie
            else:
                fraza_100 += propozitie
                embeddings = model_emb.encode(fraza_100)
                ar_interior = np.array([embeddings])

                if len(collection.get()['ids']) < 1:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[fraza_100],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=['1']
                        )
                    except ZeroDivisionError:
                        print('eroare id 1')
                else:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[fraza_100],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=[str(len(collection.get()['ids']) + 1)]
                        )
                        print('am adaugat cu succes la cu id', str(len(collection.get()['ids']) + 1), 'fraza de 100')
                    except ZeroDivisionError:
                        print('eroare id mai mare ca 1')

                # dupa ce se termina insertul in db
                fraza_100 = ''
                tokeni_100 = 0


################## pentru fraza cu 400 tokeni
            if (tokeni_400 < 400):
                tokeni_400 += nrTokeni
                fraza_400 += propozitie


            else:
                fraza_400 += propozitie
                embeddings = model_emb.encode(fraza_400)
                ar_interior = np.array([embeddings])

                if len(collection.get()['ids']) < 1:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[fraza_400],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=['1']
                        )
                    except ZeroDivisionError:
                        print('eroare id 1')
                else:
                    try:
                        collection.add(
                            embeddings=np.array(ar_interior),
                            documents=[fraza_400],
                            metadatas=[{"source": "securities_us_invest"}],
                            ids=[str(len(collection.get()['ids']) + 1)]
                        )
                        print('am adaugat cu succes la cu id', str(len(collection.get()['ids']) + 1), 'fraza de 400')
                    except ZeroDivisionError:
                        print('eroare id mai mare ca 1')


                # dupa ce se termina insertul in db
                fraza_400 = ''
                tokeni_400 = 0
