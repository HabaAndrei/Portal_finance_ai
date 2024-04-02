
import nltk
import PyPDF2
import chromadb
from nltk.tokenize import sent_tokenize
import numpy as np
from diverse import numarDeCuvinte
from InstructorEmbedding import INSTRUCTOR


client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_or_create_collection(name="carte_investitii_emb_instructor_10prop")
model_emb = INSTRUCTOR('hkunlp/instructor-base')

# client_db.delete_collection(name="carte_investitii_emb_instructor_10prop")
# print(client_db.list_collections(), 'sa vedem ce baze de date am')
# quit()

# securities_us_invest
# mutual_fonds_and_etfs



def stocamInDb(emb_ar, fraza_str):

    if numarDeCuvinte(fraza_str) <= 50  : return

    ar_interior = np.array([emb_ar])
    if len(collection.get()['ids']) < 1:
        try:
            collection.add(
                embeddings=np.array(ar_interior),
                documents=[fraza_str],
                metadatas=[{"source": "securities_us_invest"}],
                ids=['1']
            )
        except ZeroDivisionError:
            print('eroare id 1')
    else:
        try:
            collection.add(
                embeddings=np.array(ar_interior),
                documents=[fraza_str],
                metadatas=[{"source": "securities_us_invest"}],
                ids=[str(len(collection.get()['ids']) + 1)]
            )
            print('am adaugat cu succes la cu id', str(len(collection.get()['ids']) + 1))
        except ZeroDivisionError:
            print('eroare id mai mare ca 1')



###################################################################

with open('../securities_us_invest.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    nrTotalPagini = len(pdf_reader.pages)

    numar = 0
    fraza = ''

    for numarPagina in range(nrTotalPagini):
        pagina = pdf_reader.pages[numarPagina]
        text = pagina.extract_text()
        lungimeaTextului = len(text)
        # daca pagina nu are cel putin 300 de caractere sar peste ea
        if lungimeaTextului < 350:
            continue

        listaCuPropozitii = sent_tokenize(text)


        for nr_propozitie in range(len(listaCuPropozitii)):
            if(numar <= 10):
                numar+=1
                fraza += listaCuPropozitii[nr_propozitie]
            else:
                sentence = fraza
                instruction = 'Represent business paragrahp for retrive a section:'
                embeddings = model_emb.encode([[instruction, sentence]])
                stocamInDb(embeddings[0], fraza)
                fraza = ''
                numar = 0

            '''
             if nr_propozitie == len(listaCuPropozitii) - 1 and fraza :
                sentence = fraza
                instruction = 'Represent business paragrahp for retrive a section:'
                embeddings = model_emb.encode([[instruction, sentence]])
                stocamInDb(embeddings[0], fraza)
                fraza = ''
                numar = 0
            '''
