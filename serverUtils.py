try:
    import json
    import os
    import uuid
    import time
    import pandas as pd
    import uuid
    import numpy as np
    import re
    import torch
    from decouple import config
    from pymongo import MongoClient
    from pysentimiento import create_analyzer
    import elasticsearch
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup
    from elasticsearch import Elasticsearch
    from elasticsearch import helpers
    from sentence_transformers import SentenceTransformer, util
    from tqdm import tqdm
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception as e:
    print("Some Modules are Missing :{}".format(e))

# DEFINE CONSTANTS
# mongodb
MONGO_USER = config('USER')
MONGO_PASS = config('PASS')
CONNECTION_STRING = f'mongodb+srv://{MONGO_USER}:{MONGO_PASS}@cluster0.fgvaysh.mongodb.net/?retryWrites=true&w=majority'
# mongo collections
MONGO_PRODUCT_DB = 'uncoverpc'
MONGO_PRODUCT_ARTICLE_COLLECTION = 'articles'
MONGO_PRODUCT_PRODUCT_COLLECTION = 'laptops'  # TODO change to generic product
MONGO_PRODUCT_TITLE_KEY = 'Name'
MONGO_PRODUCT_LINK_KEY = 'Link'
MONGO_PRODUCT_PRICE_KEY = 'Price'
MONGO_PRODUCT_ID_KEY = '_id'
MONGO_ARTICLE_LINK_KEY = 'link'
MONGO_ARTICLE_TITLE_KEY = 'title'
MONGO_ARTICLE_DOM_KEY = 'dom'
# elastic search
ELASTIC_HOST = 'localhost'
ELASTIC_PORT = 9200
ELASTIC_SCHEME = 'http'
ELASTIC_PRODUCT_INDEX = 'productsv3'
ELASTIC_PRODUCT_PROPERTIES_KEY = 'properties'
ELASTIC_ARTICLE_INDEX = 'articles'
ELASTIC_PRODUCTID_KEY = 'ProductID'
ELASTIC_ARTICLEID_KEY = 'ArticleID'
ELASTIC_ARTICLE_SAMPLE_KEY = 'ArticleSample'
ELASTIC_ARTICLE_SAMPLE_INDEX_KEY = 'ArticleSampleIndex'
ELASTIC_ARTICLE_SAMPLE_VECTOR_KEY = 'ArticleSampleVector'
ELASTIC_ARTICLE_SAMPLE_SENTIMENT_VECTOR_KEY = 'SentimentVector'
ELASTIC_REVELVANT_ARTICLE = 'relevantArticle'
ELASTIC_ARTICLE_TITLEID_KEY = 'title'
ELASTIC_ARTICLE_LINKID_KEY = 'link'
# sentiment
SENTIMENT_NEG = 'NEG'
SENTIMENT_NEU = 'NEU'
SENTIMENT_POS = 'POS'
SENTIMENT_SCORE = 'sentimentScore'
SCORE = 'score'
# beautiful soup/webscraping
HTML_PARSER = 'html.parser'
SPLIT_BY_PERIOD = '.'
SPLIT_BY_COMMA = ','
SPLIT_NEW_LINE = '\n'
SPACE = ' '
HTML_TEXT_CONTAINER = 'p'
SPECIAL_CHARACTERS = "@^*+?_=,<>/|\"}{[]"
# AI models
SEMANTIC_SEARCH = 'msmarco-distilbert-base-v4'
SENTIMENT_ANALYSIS = 'sentiment'
SENTIMENT_LANGUAGE = 'en'
# ELASTIKNN
SIMILARITY = 'cosine'
MODEL = 'exact'
NUM_RESULT = 20


class MongoAPI(object):
    def __init__(self):
        try:
            client = MongoClient(
                CONNECTION_STRING, uuidRepresentation="standard")
            db = client[MONGO_PRODUCT_DB]
            self.articlesCollection = db[MONGO_PRODUCT_ARTICLE_COLLECTION]
            self.products = db[MONGO_PRODUCT_PRODUCT_COLLECTION]
            print("Mongo database is connected")
        except Exception as e:
            print('Could not establish a connection to mongodb')
            quit(0)

    def getArticle(self):
        return self.articlesCollection.find_one()

    def getArticles(self):
        return self.articlesCollection.find()

    def getArticleById(self, id):
        return self.articlesCollection.find({MONGO_PRODUCT_ID_KEY: uuid.UUID(id)})[0]

    def getProduct(self, id):
        return self.products.find({MONGO_PRODUCT_ID_KEY: uuid.UUID(id)})


class Tokenizer(object):
    def __init__(self):
        try:
            # init model
            self.model = SentenceTransformer(SEMANTIC_SEARCH)
            self.sentimentModel = create_analyzer(
                task=SENTIMENT_ANALYSIS, lang=SENTIMENT_LANGUAGE)
            print("Tokenizer model are done initializing")
        except Exception as e:
            print('Could not initialize Tokenizer models')
            quit(0)

    def get_token(self, documents):
        sentences = [documents]
        sentence_embeddings = self.model.encode(sentences)
        _ = list(sentence_embeddings.flatten())
        encod_np_array = np.array(_)
        encod_list = encod_np_array.tolist()
        return encod_list

    def get_sentiment_token(self, document):
        prediction = self.sentimentModel.predict(document).probas
        arr = [prediction[SENTIMENT_NEG],
               prediction[SENTIMENT_NEU], prediction[SENTIMENT_POS]]
        return np.array(arr)


class ElasticSearchImports(object):
    def __init__(self):
        start_time = time.time()
        self.es = self.connect_database()
        self.mongo = MongoAPI()
        self.tokenizer = Tokenizer()
        print("Server is ready. Initialization time: {} seconds".format(
            str(time.time()-start_time)))

    def connect_database(self):
        es = None
        es = Elasticsearch([{
            'host': ELASTIC_HOST, 'port': ELASTIC_PORT, 'scheme': ELASTIC_SCHEME
        }])
        if es.ping():
            print("Conencted to elasticsearch server")
        else:
            print(
                "A connection to Elasticsearch server could not be estabilished. Is the server on?")
            quit(0)
        return es

    def indexArticle(self, link, title, dom, relatedProduct):
        print("Indexing article: {}", link)
        artUUID = uuid.uuid4()
        self.es.index(
            index=ELASTIC_ARTICLE_INDEX,
            document={
                ELASTIC_PRODUCTID_KEY: relatedProduct,
                ELASTIC_ARTICLEID_KEY: artUUID,
                ELASTIC_ARTICLE_TITLEID_KEY: title,
                ELASTIC_ARTICLE_LINKID_KEY: link,
            }
        )
        startTime = time.time()
        # article variables
        articleSampleIndex = 0

        # html parser
        soup = BeautifulSoup(dom, HTML_PARSER)
        domain = urlparse(link).netloc
        # splitting by paragraphs
        for para in soup.find_all(HTML_TEXT_CONTAINER):

            terms = domain.split(".")
            host = ''
            if len(terms) > 2:
                host = terms[1]
            else:
                host = terms[0]
            if host in para.text:
                continue
            text = para.text
            text.replace(
                SPLIT_NEW_LINE, SPLIT_BY_PERIOD+SPACE)
            for p in text.split(SPLIT_BY_PERIOD+SPACE):
                # Filter
                if any(c in SPECIAL_CHARACTERS for c in p):
                    continue
                # Get vector mapping for each sentence
                semanticVectors = self.tokenizer.get_token(p)
                # Get sentiment vector for sentence
                sentimentVector = self.tokenizer.get_sentiment_token(p)
                # index phrase
                self.es.index(
                    index=ELASTIC_PRODUCT_INDEX,
                    document={
                        ELASTIC_PRODUCTID_KEY: relatedProduct,
                        ELASTIC_ARTICLEID_KEY: artUUID,
                        ELASTIC_ARTICLE_SAMPLE_KEY: p,
                        ELASTIC_ARTICLE_SAMPLE_INDEX_KEY: articleSampleIndex,
                        ELASTIC_ARTICLE_SAMPLE_VECTOR_KEY: semanticVectors,
                        ELASTIC_ARTICLE_SAMPLE_SENTIMENT_VECTOR_KEY: sentimentVector
                    }
                )
                articleSampleIndex = articleSampleIndex + 1
        endTime = time.time()
        print("Article has been indexed: " +
              str(endTime-startTime) + "s\n")

    def indexArticles(self, articles, relatedProduct):
        for article in articles:
            self.indexArticle(article[MONGO_ARTICLE_LINK_KEY], article[MONGO_ARTICLE_TITLE_KEY],
                              article[MONGO_ARTICLE_DOM_KEY], relatedProduct)

    def getArticleData(self, articleID):
        articleQuery = {
            'query': {
                'match': {
                    ELASTIC_ARTICLEID_KEY: {
                        'query': articleID
                    },
                }
            }
        }
        article = self.es.options().search(
            index=ELASTIC_ARTICLE_INDEX, body=articleQuery, request_timeout=55)
        articleData = [x['_source']
                       for x in article['hits']['hits']][0]
        data = {ELASTIC_ARTICLE_TITLEID_KEY: articleData[ELASTIC_ARTICLE_TITLEID_KEY],
                ELASTIC_ARTICLE_LINKID_KEY: articleData[ELASTIC_ARTICLE_LINKID_KEY]}
        return data

    def getArticleForProduct(self, product):
        for x, property in enumerate(product['properties']):
            articleQuery = {
                'query': {
                    'match': {
                        ELASTIC_ARTICLEID_KEY: {
                            'query': property[ELASTIC_ARTICLEID_KEY]
                        },
                    }
                }
            }
            article = self.es.options().search(
                index=ELASTIC_ARTICLE_INDEX, body=articleQuery, request_timeout=55)
            articleData = [x['_source']
                           for x in article['hits']['hits']][0]
            product['properties'][x]['title'] = articleData[ELASTIC_ARTICLE_TITLEID_KEY]
            product['properties'][x]['link'] = articleData[ELASTIC_ARTICLE_LINKID_KEY]
        return product

    def getIndexMatch(self, newProduct, previousProducts):
        for x, previousProduct in enumerate(previousProducts):
            if newProduct['_source'][ELASTIC_PRODUCTID_KEY] == previousProduct[ELASTIC_PRODUCTID_KEY]:
                return x
        return -1

    # TODO increase efficiency
    def rankProducts(self, newProducts, newProperty, previousProducts):
        for newProduct in newProducts:
            properties = {
                'property': newProperty,
                ELASTIC_ARTICLEID_KEY: newProduct['_source'][ELASTIC_ARTICLEID_KEY],
                ELASTIC_ARTICLE_SAMPLE_KEY: newProduct['_source'][ELASTIC_ARTICLE_SAMPLE_KEY],
                "_sentiment_score": newProduct['_sentiment_score'],
                "_relevance_score": newProduct['_score']
            }
            index = self.getIndexMatch(newProduct, previousProducts)
            if (index < 0):  # If product doesnt exist in previous data
                product = {
                    ELASTIC_PRODUCTID_KEY: newProduct['_source'][ELASTIC_PRODUCTID_KEY],
                    "_total_score": newProduct['_total_score'],
                    "_valid_properties": [newProperty],
                    'properties': [properties]
                }
                previousProducts.append(product)
            else:
                if newProperty not in previousProducts[index]['_valid_properties']:
                    previousProducts[index]['_total_score'] += newProduct['_total_score']
                    previousProducts[index]['_valid_properties'].append(
                        newProperty)
                    previousProducts[index]['properties'].append(properties)
                else:
                    previousProducts[index]['_total_score'] += newProduct['_total_score']
        return previousProducts

    def searchQuery(self, query):
        products = []
        for property in query:
            token_vector = self.tokenizer.get_token(property)
            sentiment_vector = self.tokenizer.get_sentiment_token(property)
            query = {
                "size": 20,
                "knn": {
                    "field": ELASTIC_ARTICLE_SAMPLE_VECTOR_KEY,
                    "query_vector": token_vector,
                    "k": 20,
                    "num_candidates": 100
                },
                "min_score": 0.7,
                "_source": {
                    "exclude":
                        ["ArticleSampleVector"]

                }
            }
            res = self.es.options().search(index=ELASTIC_PRODUCT_INDEX,
                                           body=query,
                                           request_timeout=55)
            data = res['hits']['hits']
            # Getting cos_sim based on sentiment
            count = 0
            for index, x in enumerate(data):

                count = count+1
                sentiment = x['_source'][ELASTIC_ARTICLE_SAMPLE_SENTIMENT_VECTOR_KEY]
                # # article = mongo.getArticleById(x['ProductID'])
                cosine_sim = cos_sim(
                    np.array(sentiment_vector), np.array(sentiment))
                x['_sentiment_score'] = cosine_sim
                x['_total_score'] = cosine_sim + x['_score']
            # Filtering out null matches
            data[:] = [x for x in data if x['_score']
                       >= 0.75 and x['_sentiment_score'] >= 0.9]
            data.sort(key=lambda x: x['_total_score'], reverse=True)
            products = self.rankProducts(data, property, products)
        products.sort(key=lambda x: x['_total_score'], reverse=True)
        # Get top three
        products = products[0:min(3, len(products))]
        for x in range(0, min(3, len(products))):
            products[x] = self.getArticleForProduct(products[x])

        return products


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
