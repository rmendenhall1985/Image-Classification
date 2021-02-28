# Image-Classification

### 1 - Redeem Your Github Student Pack
[![Github Student Pack](https://github.com/rmendenhall1985/Image-Classification/blob/main/Images/GithubStudentPack.PNG)](https://education.github.com/pack)

### 2 - Sign up for Azure for Students

[![Signup for Azure](https://github.com/rmendenhall1985/Image-Classification/blob/main/Images/SignUpForAzure.PNG?raw=true)](https://signup.azure.com/studentverification?offerType=1&correlationId=04A696E101FA66F83EE999D0002667D9)

### 3 - Brush up on Cognitive Services with Microsoft Learn

[![Launch Microsoft Learn](https://github.com/rmendenhall1985/Image-Classification/blob/main/Images/LaunchMSLearn.PNG)](https://docs.microsoft.com/en-us/learn/modules/classify-images-custom-vision/1-introduction/)

## Image Classification using Azure Cognitive Services

[Image classification](https://en.wikipedia.org/wiki/Contextual_image_classification) is a common application for [machine learning](https://en.wikipedia.org/wiki/Machine_learning). The classic use case involves training a computer to recognize cat images — or, if you're fan of the TV show [Silicon Valley](https://www.hbo.com/silicon-valley), hot-dog images. In real life, image classification serves a variety of purposes ranging from analyzing images for adult content to identifying defective parts produced by manufacturing processes. It was recently used to [help search-and-rescue drones](https://blogs.technet.microsoft.com/canitpro/2017/05/10/teaching-drones-to-aid-search-and-rescue-efforts-via-cognitive-services/) identify objects such as boats and life vests in large bodies of water and recognize potential emergency situations in order to notify a rescue squad without waiting for human intervention.

Image-classification models are typically built around [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) that "learn" from the thousands (or tens or hundreds of thousands) of labeled images they are trained with. Such networks are frequently built using [Apache Spark](https://spark.apache.org/) and [Spark ML](http://spark.apache.org/docs/latest/ml-guide.html). The [Microsoft Machine Learning Library for Apache Spark](https://github.com/Azure/mmlspark), also known as MMLSpark, is an open-source library that simplifies machine learning in Spark by abstracting many of Spark ML's lower-level APIs and providing near-seamless integration between Spark ML pipelines and popular Deep Neural Network (DNN) libraries such as the [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/), also known as the Computational Network Toolkit, or simply CNTK.

Add a file named **dedupe.py** to the project in Azure Machine Learning Workbench. Then open it for editing and paste in the following code:

```python
	# SQL Alchemy for full relational power
	# http://docs.sqlalchemy.org/en/latest/core/engines.html
	from sqlalchemy import create_engine
	import pyodbc 
	
	# Pandas for DataFrame
	# https://pypi.python.org/pypi/pandas
	import pandas as pd
	
	# Numpy
	import numpy as np
	
	# Create engine
	# http://docs.sqlalchemy.org/en/latest/dialects/mssql.html#module-sqlalchemy.dialects.mssql.pyodbc
	server = 'SERVER_NAME'
	database = 'DATABASE_NAME'
	username = 'ADMIN_USERNAME'
	password = 'ADMIN_PASSWORD'
	
	engine = create_engine('mssql+pyodbc://' + username + ':' + password + '@' + \
	    server + '.database.windows.net:1433/' + database + '?driver=ODBC+Driver+13+for+SQL+Server')
	
	# Custom SQL Query to remove duplicates
	# Solution uses T-SQL window functions https://docs.microsoft.com/en-us/sql/t-sql/queries/select-over-clause-transact-sql
	# Result set keeps the largest width image (then largest height) if the hash is equivalent
	sql = 'WITH RowTagging \
	    AS (SELECT [Artist], \
	            [ArtistNumber], \
	            [Width], \
	            [Height], \
	            [EncodingFormat], \
	            [Name], \
	            [URL], \
	            [DHashHex3], \
	            ROW_NUMBER() OVER (PARTITION BY DHashHex3 ORDER BY Width DESC, Height DESC) AS RowNumber \
	        FROM [dbo].[Paintings]) \
	    SELECT R.Artist, \
	        R.ArtistNumber, \
	        R.Width, \
	        R.Height, \
	        R.EncodingFormat, \
	        R.Name, \
	        R.URL, \
	        R.DHashHex3 \
	    FROM RowTagging R \
	    WHERE RowNumber = 1;'
	
	# Run SQL Query in SQL Azure, return results into Pandas DataFrame
	# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql.html
	df = pd.read_sql(sql, engine)
	print('Columns: ', list(df.columns.values))
	print('DataFrame Shape: ', df.shape)
	
	# Assign a random column using numpy
	df['Random'] = np.random.rand(df.shape[0])
	
	# Output Pandas DataFrame to Azure SQL
	# Note that the output would add an index by default
	# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html
	df.to_sql('UniquePaintings', engine, if_exists='replace', index=False)
```

[![Deploy to Azure](https://github.com/rmendenhall1985/SummitDemo/blob/main/Images/DeploytoAzure.PNG)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure%2Fazure-quickstart-templates%2Fmaster%2F101-cognitive-services-Computer-vision-API%2Fazuredeploy.json)

In this lab, the first of four in a series, you will use the Azure CLI to create an Azure SQL database in the cloud. Then you will use [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation) to run a Python script that uses the [Bing Image Search API](https://azure.microsoft.com/services/cognitive-services/bing-image-search-api/) to search the Web for images of paintings by famous artists and write the results to the Azure SQL database. In [Lab 2](../2%20-%20Process), you will clean the data, and in [Lab 3](../3%20-%20Predict), you will use MMLSpark to build an image-classification model that can identify the artists of famous paintings. Finally, in [Lab 4](../4%20-%20Visualize), you will operationalize the model and build an app that uses it.
