
import chromadb

client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_or_create_collection(name="carte_investitii_emb_instructor_10prop")

arCuDoc = collection.get()['documents']

def scrieInFiserText(ar):
    with open('./doc_din_db.txt', 'w', encoding='utf-8') as file:

        for fraza in ar:
            file.write('\n\n' + fraza)

scrieInFiserText(arCuDoc)
