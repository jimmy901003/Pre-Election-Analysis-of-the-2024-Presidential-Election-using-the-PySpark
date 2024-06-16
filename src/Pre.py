import numpy as np
import pandas as pd
import findspark
findspark.init()
from pyspark.sql.functions import explode, col, from_json, when, countDistinct,  regexp_extract
from pyspark.sql.types import *

from pyspark.sql.functions import explode, col, from_json, split

from pyspark.sql import SparkSession
# 創建SparkSession,並設置一些優化配置
spark = SparkSession.builder \
    .appName("preprocessing with PySpark") \
    .config("spark.sql.files.maxPartitionBytes", 512*1024*1024) \
    .config("spark.sql.files.openCostInBytes", 1024*1024*1024) \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.memory", "12g")\
    .config("spark.executor.instances", "4") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()
    
# 讀取CSV
file_path = r"D:/project/Pre-Election-Analysis-of-the-2024-Presidential-Election-using-the-PySpark-Framework/hatepolitics.csv"
df = pd.read_csv(file_path, encoding='utf-8')

df = spark.createDataFrame(df)
df = df.dropDuplicates()


# 將字串欄位解析為陣列
comments_array = from_json(col("所有留言"), ArrayType(StructType([
    StructField("type", StringType(), True),
    StructField("user", StringType(), True),
    StructField("content", StringType(), True),
    StructField("ipdatetime", StringType(), True)
])))
    
# 將陣列展開成新的資料列 
comment_df = df.select("*", explode(comments_array).alias("comment"))
comment_df = comment_df.select("*", "comment.*")
comment_df.select("ipdatetime").show(truncate = 30)

# 正則表達提取 ip及日期
comment_df = comment_df.withColumn("ip", regexp_extract("ipdatetime", r"(\d+\.\d+\.\d+\.\d+)", 1)) \
                         .withColumn("date", regexp_extract("ipdatetime", r"(\d{2}/\d{2})", 1))

comment_df.select("ip", "date").show()
             
# 刪除原始的ipdatetime欄位
comment_df = comment_df.drop("ipdatetime")
comment_df.show()
comment_df.select("date").show(truncate = 30)

from pyspark.sql.functions import col, to_date, concat, lit, month

# 將日期欄位解析為日期對象
comment_df = comment_df.withColumn("date_parsed", to_date(col("date"), "MM/dd"))

# 添加一個新的欄位來保存調整後的日期
comment_df = comment_df.withColumn("adjusted_year", 
                                   when(month("date_parsed") == 1, lit(2024))
                                   .otherwise(lit(2023)))

# 構建新的日期欄位，將調整後的年份與原始月份和日期合併
comment_df = comment_df.withColumn("adjusted_date", 
                                   concat(col("adjusted_year"), lit("-"), month("date_parsed"), lit("-"), col("date_parsed").substr(9, 2)))

comment_df = comment_df.withColumn('adjusted_date', to_date(col('adjusted_date'), 'yyyy-M-d'))

# 顯示調整後的日期欄位
comment_df.printSchema()
comment_df.select("adjusted_date").show()

from pyspark.sql.functions import udf
import geoip2.database

# GeoLite2 數據庫文件路徑
db_file = "D:\\project\\Pre-Election-Analysis-of-the-2024-Presidential-Election-using-the-PySpark-Framework\\GeoLite2-City.mmdb"

# 定義UDF以查詢IP地址所在城市
def get_city(ip):
    try:
        reader = geoip2.database.Reader(db_file)
        response = reader.city(ip)
        return response.city.name
    except:
        return None

# 將UDF註冊為Spark函數
get_city_udf = udf(get_city, StringType())

# 新增城市欄位
comment_df = comment_df.withColumn("city", get_city_udf("ip"))

comment_df.show()

data_length = comment_df.count()
print("資料的長度為:", data_length)

comment_df = comment_df.drop("所有留言").drop("comment")

from pyspark.sql.functions import when

comment_df = comment_df.withColumn(
    "content",
    when(comment_df["content"].contains("賴功德") | 
         comment_df["content"].contains("賴蕭配") | 
         comment_df["content"].contains("綠營") | 
         comment_df["content"].contains("賴蕭") | 
         comment_df["content"].contains("賴皮") | 
         comment_df["content"].contains("賴總統") | 
         comment_df["content"].contains("賴神") | 
         comment_df["content"].contains("穆罕清德") | 
         comment_df["content"].contains("台南賴") | 
         comment_df["content"].contains("賴半天") | 
         comment_df["content"].contains("賴彈性") | 
         comment_df["content"].contains("台獨金孫") | 
         comment_df["content"].contains("賴桑") | 
         comment_df["content"].contains("賴導") | 
         comment_df["content"].contains("民進黨") | 
         comment_df["content"].contains("瞑進黨") | 
         comment_df["content"].contains("Lie 神"),
         "賴清德"
    ).otherwise(comment_df["content"])
)

comment_df = comment_df.withColumn(
    "content",
    when(comment_df["content"].contains('柯P') | 
         comment_df["content"].contains('柯盈配') | 
         comment_df["content"].contains('柯盈') | 
         comment_df["content"].contains('柯總統') | 
         comment_df["content"].contains('民眾黨') | 
         comment_df["content"].contains('可達鴉') | 
         comment_df["content"].contains('阿北') | 
         comment_df["content"].contains('柯p') | 
         comment_df["content"].contains('科文哲') | 
         comment_df["content"].contains('蚵蚊蜇') | 
         comment_df["content"].contains('亞斯伯哲') | 
         comment_df["content"].contains('柯師父') | 
         comment_df["content"].contains('柯seafood') | 
         comment_df["content"].contains('柯阿北') | 
         comment_df["content"].contains('火影柯'),
         '柯文哲'
    ).otherwise(comment_df["content"])
)

comment_df = comment_df.withColumn(
    "content",
    when(comment_df["content"].contains('侯副') | 
         comment_df["content"].contains('侯康配') | 
         comment_df["content"].contains('侯康') | 
         comment_df["content"].contains('國民黨') | 
         comment_df["content"].contains('藍營') | 
         comment_df["content"].contains('猴康配') | 
         comment_df["content"].contains('侯老三') | 
         comment_df["content"].contains('猴猴') | 
         comment_df["content"].contains('侯侯') | 
         comment_df["content"].contains('猴老三') | 
         comment_df["content"].contains('侯總統') | 
         comment_df["content"].contains('猴總統') | 
         comment_df["content"].contains('阿猴') | 
         comment_df["content"].contains('封城侯') | 
         comment_df["content"].contains('猴友誼') | 
         comment_df["content"].contains('侯Sir') | 
         comment_df["content"].contains('侯導') | 
         comment_df["content"].contains('猴導') | 
         comment_df["content"].contains('hohoGPT') | 
         comment_df["content"].contains('HoHo GPT'),
         '侯友宜'
    ).otherwise(comment_df["content"])
)

from pyspark.sql.functions import col, sum

# 計算每列空值數量
null_counts_exprs = [sum(col(column).isNull().cast("int")).alias(f"{column}_null_count") for column in comment_df.columns]

# 計算及果轉為pd dataframe
null_counts_df = comment_df.agg(*null_counts_exprs).toPandas()
pd.set_option('display.max_columns', None)

print(null_counts_df)

# 過濾空字串並計算數量
empty_content_count = comment_df.filter(col("content") == "").count()

print("content列中的空字串數量:", empty_content_count)

# 過濾content不為空
comment_df= comment_df.filter(comment_df["content"] != "")

# unique_dates = comment_df.select("adjusted_date").distinct()
# # 顯示唯一值
# unique_dates.show()
# comment_df.select("adjusted_date").show()
# comment_df.printSchema()


comment_df = comment_df.filter((col("adjusted_date") >= "2023-12-31") & (col("adjusted_date") <= "2024-01-12"))

comment_df.write.mode("overwrite").parquet('comment_df')
com_comment_df = comment_df.toPandas()
com_comment_df.to_csv("com_comment_df.csv", index=False)
spark.stop()