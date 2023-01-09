from flask import Flask, request, jsonify
from serverUtils import ElasticSearchImports
import time
app = Flask('uncoverpc-searchengine')

route = '/api'

esi = ElasticSearchImports()


@app.route(route + '/addArticles', methods=['POST'])
def indexArticle():
    print('indexing articles')


@app.route(route + '/searchProduct', methods=['POST'])
def searchProduct():
    start = time.time()
    query = request.json
    # only in form of array
    # query = ['data']
    product = esi.searchQuery(query)
    print(str(time.time()-start))
    return jsonify(product)

# @app.route(route + '/s')


app.run()


# start = time.time()
# product = esi.searchQuery(['good battery life', 'good vibrant display'])
# print(str(time.time()-start))
# print(product)
# TEST
# documents = esi.mongo.getArticles()
# for doc in documents:
#     esi.indexArticles(doc['Articles'], doc['_id'])
# print("DONE")
