from openai import OpenAI
from sklearn.metrics.pairwise import  cosine_similarity
import os
load_dotenv()

OPENAI_key = os.getenv('OPENAI_key')

client = OpenAI(api_key=OPENAI_key)

#EMBADDING
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#df = pd.DataFrame()
#df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

input_list = [

   ''''  În primele 9 luni ale anului 2023, preţul de tranzacţionare al acţiunilor Romgaz a oscilat între o
   valoare minimă de 38 lei (atinsă în 24 martie 2023) şi o valoare maximă de 45 lei (realizată în ultima
   zi de tranzacţionare, respectiv 29 septembrie 2023). Valoarea medie a acţiunii a fost de 40,56 lei,
   preţul acţiunii înregistrând o creştere la sfârşitul lunii septembrie comparativ cu sfârşitul anului
   2022 de 19,21%, iar comparativ cu aceeaşi perioadă a anului trecut de 12,50%. Totodată, preţul
   mediu al acţiunii a crescut trimestrial, de la 40,11 lei în T1 2023, la 40,17 lei în T2 2023 şi, respectiv,
   la 41,31 lei în T3 2023. ''',
   '''Evoluţia preţului acţiunii în perioada ianuarie – septembrie 2023 a fost oscilantă, remarcându-se
   următoarele evenimente principale care au determinat scăderi şi creşteri accentuate: aprobarea de
   către Guvernul României a unui memorandum pentru repartizarea unei cote de minim 90% din
   profitul net realizat de societăţile cu capital majoritar de stat în anul 2022 sub formă de dividende
   (creştere: începutul lunii martie 2023), scăderea preţurilor de referinţă la gazele din Europa sub
   pragul de 40 EUR (43 USD) pentru un Mwh, în condiţiile în care vremea blândă din iarna
   2022/2023 a redus cererea4
   (scădere: sfârşitul lunii martie 2023), creşterea interesului investitorilor
   pentru acţiunile tranzacţionate la Bursa de Valori Bucureşti (BVB) ca urmare a luării deciziei finale
   de investiţie pentru proiectul Neptun Deep şi aprobarea planului de dezvoltare de către OMV
   Petrom şi Romgaz, precum şi a demarării procesului de listare la bursă a societăţii Hidroelectrica
   (creştere: sfârşitul lunii iunie 2023), ex-data dividende 2022 şi publicarea indicatorilor cheie pe
   semestrul 1 2023 care evidenţiază scăderea producţiei de hidrocarburi (scădere în luna iulie 2023),
   scăderi accentuate ale indicilor BVB în ton cu deprecierile pieţelor europene, ca urmare a temerilor
   investitorilor la nivel mondial cu privire la evoluţia pieţelor 5
   (scăderi: luna august 2023),
   perspectivele bune pe termen lung ale companiei, având în vedere proiectul de exploatare a gazelor
   în Marea Neagră6
   (creştere: începutul lunii septembrie 2023), anunţarea unui contract de cca 1
   mld.lei cu E.ON Energie Romania7 (creştere: sfârşitul lunii septembrie 2023).''',

   '''Preţul certificatelor de depozit globale (GDR), care au la bază acţiuni Romgaz, a avut o evoluţie
   uşor diferită faţă de cea a acţiunilor, astfel că preţul minim al perioadei a fost înregistrat în data de
   19 mai 2023: 7,95 USD (echivalent 36,63 lei), iar preţul maxim a fost atins în 25 iulie 2023: 11,10
   USD (49,44 lei).
   Preţul mediu de tranzacţionare a GDR-urilor a fost 8,93 USD (echivalentul a 40,70 lei) în cele 9
   luni ale anului 2023, cu valoarea cea mai mare înregistrată tot în T3 2023, similar acţiunilor: 9,33
   USD (echivalent 42,40 lei). Luna septembrie s-a încheiat cu preţul de 8,55 USD/GDR (echivalent
   40,07 lei/GDR), în creştere faţă de sfârşitul anului cu 6,88% în USD, respectiv cu 8,07% în lei (ca
   urmare a creşterii cursului de schimb USD/lei cu 1,12%).
   '''
]

input_embeddings = [ get_embedding(sentence) for sentence in input_list];
question = 'Care a fost pretul mediu al actiunii romgaz '
question_embeddings = get_embedding(question)


# COSINE
similarity_scores = [cosine_similarity([question_embeddings], [p])[0][0] for p in input_embeddings]
ranked_paragraphs = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
top_paragraph_index, top_similarity_score, = ranked_paragraphs[0]


# COMPLETION
print(input_list, ' ----------  contextul ----------')
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": 'context : ' + input_list[top_paragraph_index] + 'question: ' + question}
  ]
)

print(completion.choices[0].message.content)
