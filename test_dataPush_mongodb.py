'''This python file was created to test mongdb cluster connection with our application.
You can test it by running python {this_file_name}.py'''

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
# Load environment variables from the .env file
load_dotenv()


# Access the URI value
url = os.getenv("MONGO_DB_URL")

password = os.getenv("MONGO_DB_PASS")
Username = os.getenv("MONGO_DB_UserName")

uri = f"mongodb+srv://{Username}:{password}@networksecuritycluster.zn3c0.mongodb.net/?retryWrites=true&w=majority&appName=NetworkSecurityCluster"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)