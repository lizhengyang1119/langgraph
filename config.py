"""
1.负责模型实例化
2.负责从.env文件读取配置
"""


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载当前目录 .env 文件
load_dotenv()


model = ChatOpenAI(
    model="deepseek-v4-flash",  # 复杂任务使用：deepseek-v4-pro，deepseek-v4-flash 
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com",
    extra_body={"thinking": {"type": "disabled"}} # 关闭思考模式
)

# 读取天气 API密钥
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
# 读取Tavily API密钥
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")