
# =>>> env
################################################################


import chromadb
from openai import OpenAI
from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np
from diverse import scriem_in_fiser, scriem_in_fiserMesAi
from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_key = os.getenv('OPENAI_key')

client_db = chromadb.HttpClient(host="localhost", port=8000)
collection = client_db.get_collection(name="carte_investitii_emb_instructor_10prop")
client_ai = OpenAI(api_key=OPENAI_key)
model_emb = INSTRUCTOR('hkunlp/instructor-base')



app = Flask(__name__)
CORS(app)


def functieGeneratore(mes):


    scriem_in_fiserMesAi(mes)

    completion = client_ai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=mes,
        stream=True,
        temperature=0.5,
        frequency_penalty=1.8
    )

    # print(mes)

    for chunk in completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content.encode('utf-8')





@app.route("/cerereAI" , methods = ['POST'])
def index():

    data = request.json
	
    


    intrebare = data['intrebare']
    context = data['context']

    #print('---------------------------', intrebare , context  ,'----------------------')

    intrebare_emb = model_emb.encode('Question:' +  intrebare)


    ar_interior = np.array([intrebare_emb])
    results = collection.query(
        query_embeddings=np.array(ar_interior),
        n_results=5
    )


    # print(results['distances'][0], results['documents'][0])

    frazaCuRaspunsuri = ''
    for nrProp in range(len(results['documents'][0])):
        frazaCuRaspunsuri += '\n\n' + 'Answer ' + str(nrProp + 1) + ': ' + results['documents'][0][nrProp]

    if(results['documents'][0]): scriem_in_fiser(intrebare, results['documents'][0])

    mes = [
        {"role": "system",
         "content": "You are a robot operating in the financial domain, always be sure of what you say, but guide the person to do further research, and the response should be medium and to the point. The most important is question. "},
        {"role": "user", "content": 'context : ' + frazaCuRaspunsuri + '       question: ' + intrebare},

    ]


    for obiect in context:
        ob = {}
        if obiect['tip_mesaj'] == 'raspuns':
            ob['role'] = 'assistant'
            ob['content'] = obiect['mesaj']
            mes.append(ob)
        elif obiect['tip_mesaj'] == 'intrebare':
            ob['role'] = 'user'
            ob['content'] = obiect['mesaj']
            mes.append(ob)

        # ob['role'] = 'assistant'




    return Response(functieGeneratore(mes), mimetype = "text/event-stream" )




if __name__ == '__main__':
    app.run(host='195.201.17.190', port=4000, debug=True)

