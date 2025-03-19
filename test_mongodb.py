from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi  # Import certifi to handle SSL certificates

# MongoDB Atlas connection URI
uri = "mongodb+srv://adityasinha3107:Admin123@cluster0.kto7g.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(
    uri,
    server_api=ServerApi('1'),
    tls=True,  # Enable TLS/SSL
    tlsCAFile=certifi.where()  # Use certifi's CA bundle
)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)