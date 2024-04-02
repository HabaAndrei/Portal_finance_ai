import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import  cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_key = os.getenv('OPENAI_key')


client = OpenAI(api_key=OPENAI_key)
# client_db = chromadb.HttpClient(host="localhost", port=8000)

##########################
###### Impartim fraza in propozitii

# fraza =  '''
#     Please feel free to contact us with any of your questions or concerns about investing. It always pays to learn before you invest. And congratulations on taking your first step on the road to financial security.
#     '''
# listaCuFraze = sent_tokenize(fraza)

###########################
####### Cream embeddingurile pentru fiecare propozitie

# model = SentenceTransformer('thenlper/gte-small')
# embeddings = model.encode(listaCuFraze)
#########################################
########### Adaugam in baza de date

# adaug in db pentru fiecare embedding
# collection = client_db.get_or_create_collection(name="colectie_fraze_emb")
#
#
# for enb, prop in zip(embeddings, listaCuFraze):
#     ar_interior = np.array([enb])
#     if len(collection.get()['ids']) < 1:
#         collection.add(
#             embeddings=np.array(ar_interior),
#             documents=[prop],
#             metadatas=[{"source": "my_source"}],
#             ids=['1']
#         )
#     else :
#         collection.add(
#             embeddings=np.array(ar_interior),
#             documents=[prop],
#             metadatas=[{"source": "my_source"}],
#             ids=[str(len(collection.get()['ids']) + 1)]
#         )

# client_db.delete_collection(name="colectie_fraze_emb")
# collection = client_db.get_collection(name="colectie_fraze_emb")
# print(collection.get(include=['embeddings']))

# results = collection.query(
#     query_texts=["How can you avoid investment fraud and costly mistakes?"],
#     n_results=2
# )
# print(results)

# print(collection.get())
# print(client_db.list_collections())





