from flask import Flask, request, jsonify
from serverUtils import ElasticSearchImports
import time
app = Flask(__name__)

route = '/api'

esi = ElasticSearchImports()


@app.route(route + '/addArticles', methods=['POST'])
def indexArticle():
    print('indexing articles')


@app.route(route + '/searchProduct', methods=['POST'])
def searchProduct():
    start = time.time()
    query = request.json
    product = esi.searchQuery(query)
    print(str(time.time()-start))
    return jsonify(product)


@app.route('/', methods=['GET'])
def index():
    # print(esi.es.cluster.health())
    health = esi.es.cluster.health()
    data = {
        'Search Engine Server Status': 'Working',
        'Server': 'UncoverPC-SearchEngine',
        'Region': 'Local',
        'ElasticSearch Cluster': {"Heath": health['status'], 'number_of_nodes': health['number_of_nodes']}

    }
    return data


if __name__ == "__main__":
    app.run(debug=True)

# TEST - To index articles stored in mongodb
# documents = esi.mongo.getArticles()
# for doc in documents:
#     esi.indexArticles(doc['Articles'], doc['_id'])
# print("DONE")
