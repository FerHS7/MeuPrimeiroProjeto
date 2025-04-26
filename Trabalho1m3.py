from IPython import get_ipython
import prettytable
import sqlite3
import pandas as pd

get_ipython().run_line_magic('load_ext', 'sql')
prettytable.DEFAULT = 'DEFAULT'
con =sqlite3.connect("my_data1.db")
con = con.cursor()
from sqlalchemy import create_engine

engine = create_engine('sqlite:///my_data1.db')
connection = engine.connect()
import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")
#DROP THE IF EXISTS
con.execute("DROP TABLE IF EXISTS SPACEXTABLE;")
con.execute("CREATE TABLE SPACEXTABLE AS SELECT * FROM SPACEXTBL WHERE Date IS NOT NULL;")
