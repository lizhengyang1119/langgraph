from langchain.tools import tool
from langchain_openai import ChatOpenAI
# 导入模型实例
from config import model


# 把函数定义成一个工具
@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积 a * b。"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和 a + b。"""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """计算 a 除以 b 的商（浮点数）。"""
    return a / b


# 增强LLM的工具能力
tools = [add, multiply, divide]
# 给工具起名字
tools_by_name = {tool.name: tool for tool in tools}
# 调用 llm 时，除了传入对话消息，还会附带 「可用工具列表」（工具名称 + 使用说明：也就是函数中的注释 + 参数格式：由@tool自动生成）
model_with_tools = model.bind_tools(tools)


from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

# 定义状态（包含：对话消息列表，调用 LLM 的次数）
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # operator.add用于拼接新消息和旧消息
    llm_calls: int  # 每执行一次「模型节点」里的 invoke，就把计数加 1。


#-----------------------------------------以上定义了模型，工具，状态-----------------------------------------


#-----------------------------------------下面定义了模型节点，工具节点-----------------------------------------

from langchain.messages import SystemMessage

# 调用llm_call，返回的是一个状态。
def llm_call(state: dict):
    """LLM决定是否调用工具"""

    return {
        # 返回的消息字典中会包含有关键字参数：tool_calls
        "messages": [model_with_tools.invoke([SystemMessage(content="你是一个有用的助手，负责对一组输入执行算术运算。")]+ state["messages"])],
        "llm_calls": state.get('llm_calls', 0) + 1  # 读出状态中llm调用次数，没有就当成0，然后+1
    }



from langchain.messages import ToolMessage


def tool_node(state: dict):
    """执行工具调用"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

#-----------------------------------------下面定义了结束逻辑-----------------------------------------

from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """决定是否继续循环或停止，基于LLM是否进行了工具调用"""

    messages = state["messages"]
    # 提取合并状态中的消息列表之后，排在最末尾的那条刚挂上的消息。（可以去看一下大模型返回值，就明白了）
    last_message = messages[-1]

    # 如果LLM进行了工具调用，则执行操作
    if last_message.tool_calls:
        return "tool_node"

    # 否则，我们停止（回复用户）
    return END


#-----------------------------------------下面构建工作流-----------------------------------------


# 初始化构图对象
agent_builder = StateGraph(MessagesState)

# 添加模型节点
agent_builder.add_node("llm_call", llm_call)

# 添加工具节点
agent_builder.add_node("tool_node", tool_node)

# 添加固定边
agent_builder.add_edge(START, "llm_call")

# 添加固定边
agent_builder.add_edge("tool_node", "llm_call")

# 添加条件边（不确定去那个节点，而是由函数 should_continue 根据当前状态决定走哪条路）
agent_builder.add_conditional_edges( 
    "llm_call",  # 起始节点
    should_continue,  # 决策函数，会return目的节点
    ["tool_node", END]  # 共有两个目的节点
)


# 把前面搭好的图「编译」成可以真正运行的对象，并赋给变量 agent
agent = agent_builder.compile()

# 调用
from langchain.messages import HumanMessage

messages = [HumanMessage(content="3加4等于多少。")]

# 这里的invoke是让agent运行，不是调用llm的意思,返回的是最终的状态
messages = agent.invoke({"messages": messages})


for m in messages["messages"]:
    # 打印消息，美化输出
    m.pretty_print()



"""
用于可视化你构建的图
"""
from PIL import Image
import io
# 获取 PNG 二进制数据
png_data = agent.get_graph().draw_mermaid_png()
# 转换为 PIL Image 对象并显示
img = Image.open(io.BytesIO(png_data))
img.show()   # 会用系统默认图片查看器打开

