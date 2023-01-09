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
MONGO_PRODUCT_DB = 'Products'
MONGO_PRODUCT_ARTICLE_COLLECTION = 'Articles'
MONGO_PRODUCT_PRODUCT_COLLECTION = 'Products'
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
ELASTIC_PRODUCT_INDEX = 'products'
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
            if para.text.count(SPLIT_BY_PERIOD) < 3:
                continue
            text = para.text
            text.replace(
                SPLIT_NEW_LINE, SPLIT_BY_PERIOD+SPACE)
            for p in text.split(SPLIT_BY_PERIOD+SPACE):
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

    def getTopProduct(self, products):
        tops = None
        count = 0
        for product in products:
            if count == 0:
                tops = product
                count = count + 1
            if (products[product][SCORE] > products[tops][SCORE]):
                tops = product
        product = self.mongo.getProduct(tops)[0]
        result = {
            'Product': products[tops],
            MONGO_ARTICLE_TITLE_KEY: product[MONGO_PRODUCT_TITLE_KEY],
            MONGO_PRODUCT_LINK_KEY: product[MONGO_PRODUCT_LINK_KEY],
            MONGO_PRODUCT_PRICE_KEY: product[MONGO_PRODUCT_PRICE_KEY]
        }
        return result

# TODO fix algorithm
    def rankProducts(self, products, property, index, productId, count):
        id = productId
        if id in products:
            # Adding score
            if property in products[id][ELASTIC_PRODUCT_PROPERTIES_KEY]:
                products[id][ELASTIC_PRODUCT_PROPERTIES_KEY][property][SENTIMENT_SCORE] = products[id][ELASTIC_PRODUCT_PROPERTIES_KEY][property][SCORE] + \
                    index[ELASTIC_REVELVANT_ARTICLE][SENTIMENT_SCORE]
                products[id][SCORE] = products[id][SCORE] + \
                    index[SCORE]
            else:
                products[id][ELASTIC_PRODUCT_PROPERTIES_KEY][property] = index
                products[id][SCORE] = products[id][SCORE] + \
                    index[SCORE]
            # Adding relevant articles -> contains link, title, and relevant article snippet
            products[id][ELASTIC_PRODUCT_PROPERTIES_KEY][property] = index
        else:
            product = {
                SCORE: index[SCORE],
                ELASTIC_PRODUCT_PROPERTIES_KEY: {
                    property: index
                }
            }
            products[id] = product

        return products

    def searchQuery(self, query):
        products = {}
        for property in query:
            token_vector = self.tokenizer.get_token(property)
            sentiment_vector = self.tokenizer.get_sentiment_token(property)
            query = {
                "query": {
                    "elastiknn_nearest_neighbors": {
                        "vec": token_vector,
                        "field": ELASTIC_ARTICLE_SAMPLE_VECTOR_KEY,
                        "similarity": SIMILARITY,
                        "model": MODEL,
                    }
                },
                'size': NUM_RESULT
            }
            res = self.es.options().search(index=ELASTIC_PRODUCT_INDEX,
                                           body=query,
                                           request_timeout=55)
            data = [x['_source'] for x in res['hits']['hits']]
            # Getting cos_sim based on sentiment
            count = 0
            for x in data:
                count = count+1
                function = (-1/NUM_RESULT * count + 1)
                sentiment = x[ELASTIC_ARTICLE_SAMPLE_SENTIMENT_VECTOR_KEY]
                # article = mongo.getArticleById(x['ProductID'])
                cosine_sim = cos_sim(
                    np.array(sentiment_vector), np.array(sentiment))

                index = {
                    SCORE: function + cosine_sim,
                    ELASTIC_REVELVANT_ARTICLE: {
                        'title': '',
                        'link': '',
                        'articleSample': x[ELASTIC_ARTICLE_SAMPLE_KEY],
                        'boldStart': '',
                        'boldEnd': '',
                        SENTIMENT_SCORE: cosine_sim
                    }
                }
                products = self.rankProducts(
                    products, property, index, x[ELASTIC_PRODUCTID_KEY], function)

        product = self.getTopProduct(products)
        return product


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
