import psycopg2
import json

hostname = 'localhost'
username = ''
password = ''
database = 'machinelearning'

# Simple routine to run a query on a database and print the results:
def addRow(conn=None, image=None, _class=None, score=None, embeddings=[], group=None):
    if conn is None or image is None or _class is None or score is None or embeddings == [] or group is None:
        print("Missing one element in query")
        return

    cur = conn.cursor()
    emb = "{" + ",".join([str(i) for i in embeddings]) + "}"
    sql = "INSERT INTO inference (image,class,score,embeddings,group_name) VALUES (%s,%s,%s,%s,%s)"
    cur.execute(sql, (image, _class, score, emb, group))
    conn.commit()


# def getAllResults(conn) :
#     cur = conn.cursor()
#     cur.execute( "SELECT * FROM inference" )
#
#     for res in cur.fetchall():
#         print(res)


conn = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )


group = 5
with open(f'/Users/i337936/Downloads/inference/{group}.json') as data_file:
    print("Loading Json...")
    data = json.load(data_file)

    print("Importing data...")
    for d in data:
        name = d[0].replace("/ib/junk/junk/shany_ds/shany_proj/dataset/inference/","")
        _class = d[1][0]
        score = d[1][1]
        embeddingData = d[2]
        addRow(conn, name, _class, score, embeddingData, group)

print('Import complete')
conn.close()