#%%
# Import required modules and initialize the builder from open_deep_research
import uuid 
import os, getpass
import open_deep_research   
print(open_deep_research.__version__) 
from IPython.display import Image, display, Markdown
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
#%%
# 셀 1: 원본 함수 패치 (가장 확실한 방법)
import open_deep_research.graph as graph_module
from langgraph.types import Command
from langgraph.constants import Send

# 원본 함수 백업
original_human_feedback = graph_module.human_feedback

def patched_human_feedback(state, config):
    """자동 승인하는 패치된 human_feedback"""
    topic = state["topic"]
    sections = state['sections']
    
    print(f"🤖 자동 승인: {len(sections)}개 섹션 중 {len([s for s in sections if s.research])}개가 연구 필요")
    
    # 원본과 정확히 동일한 승인 로직
    return Command(goto=[
        Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
        for s in sections 
        if s.research
    ])

# 함수 교체
graph_module.human_feedback = patched_human_feedback

print("✅ human_feedback 함수가 자동 승인으로 패치되었습니다!")

#%%
# 강력한 패치 방법
import open_deep_research.graph

# 모듈 전체를 다시 로드
import importlib
importlib.reload(open_deep_research.graph)

# 다시 패치
def auto_approve_only(state, config):
    """무조건 자동 승인"""
    topic = state["topic"]
    sections = state['sections']
    
    print(f"🚀 무조건 자동 승인! 연구 섹션: {len([s for s in sections if s.research])}개")
    
    return Command(goto=[
        Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
        for s in sections 
        if s.research
    ])

# 강제 교체
open_deep_research.graph.human_feedback = auto_approve_only

# 그래프 재생성
from open_deep_research.graph import builder
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

#%%
# Create a memory-based checkpointer and compile the graph
# This enables state persistence and tracking throughout the workflow execution

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
#%%
# Visualize the graph structure
# This shows the nodes and edges in the research workflow

display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
#%%
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 값이 로드되었는지 확인
required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", 
                "GROQ_API_KEY", "PERPLEXITY_API_KEY"]

missing_keys = []
for key in required_keys:
    if not os.getenv(key):
        missing_keys.append(key)

if missing_keys:
    print(f"Missing API keys: {missing_keys}")
    print("Please check your .env file")
else:
    print("✅ All API keys loaded successfully!")
#%%
# # Helper function to set environment variables for API keys
# # This ensures all necessary credentials are available for various services
# 
# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# 
# # Set the API keys used for any model or search tool selections below, such as:
# _set_env("OPENAI_API_KEY")
# _set_env("ANTHROPIC_API_KEY")
# _set_env("TAVILY_API_KEY")
# _set_env("GROQ_API_KEY")
# _set_env("PERPLEXITY_API_KEY")
#%%
# Define report structure template and configure the research workflow
# This sets parameters for models, search tools, and report organization

REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

# Configuration option 1: Claude 3.7 Sonnet for planning with perplexity search
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "perplexity",
                           "planner_provider": "anthropic",
                           "planner_model": "claude-3-7-sonnet-latest",
                           # "planner_model_kwargs": {"temperature":0.8}, # if set custom parameters
                           "writer_provider": "anthropic",
                           "writer_model": "claude-3-5-haiku-latest",
                           # "writer_model_kwargs": {"temperature":0.8}, # if set custom parameters
                           "max_search_depth": 2,
                           "report_structure": REPORT_STRUCTURE,
                            "skip_human_feedback": True,  # 이런 옵션 찾아보기
                            "auto_approve": True,         # 또는 이런 옵션
                           }}

# Configuration option 2: DeepSeek-R1-Distill-Llama-70B for planning and llama-3.3-70b-versatile for writing
# thread = {"configurable": {"thread_id": str(uuid.uuid4()),
#                            "search_api": "tavily",
#                            "planner_provider": "groq",
#                            "planner_model": "deepseek-r1-distill-llama-70b",
#                            "writer_provider": "groq",
#                            "writer_model": "llama-3.3-70b-versatile",
#                            "report_structure": REPORT_STRUCTURE,
#                            "max_search_depth": 1,}
#                            }

# Configuration option 3: Use OpenAI o3 for both planning and writing (selected option)
# thread = {"configurable": {"thread_id": str(uuid.uuid4()),
#                            "search_api": "tavily",
#                            "planner_provider": "openai",
#                            "planner_model": "o3",
#                            "writer_provider": "openai",
#                            "writer_model": "o3",
#                            "max_search_depth": 2,
#                            "report_structure": REPORT_STRUCTURE,
#                            }}

# Define research topic about Model Context Protocol
topic = "Overview of Model Context Protocol (MCP), an Anthropic‑backed open standard for integrating external context and tools with LLMs. Give an architectural overview for developers, tell me about interesting MCP servers, and compare to google Agent2Agent (A2A) protocol."

# Run the graph workflow until first interruption (waiting for user feedback)
async for event in graph.astream({"topic":topic,}, thread, stream_mode="updates"):
    if '__interrupt__' in event:
        interrupt_value = event['__interrupt__'][0].value
        display(Markdown(interrupt_value))

#%%
# # Submit feedback on the report plan
# # The system will continue execution with the updated requirements
# 
# # Provide specific feedback to focus and refine the report structure
# async for event in graph.astream(Command(resume="Looks great! Just do one section related to Agent2Agent (A2A) protocol, introducing it and comparing to MCP."), thread, stream_mode="updates"):
#     if '__interrupt__' in event:
#         interrupt_value = event['__interrupt__'][0].value
#         display(Markdown(interrupt_value))
#%%
# human_feedback를 건너뛰고 바로 research 단계로
async for event in graph.astream(Command(goto="generate_queries"), thread, stream_mode="updates"):
    print(event)
#%%
async for event in graph.astream(Command(resume="true"), thread, stream_mode="updates"):
    if '__interrupt__' in event:
        interrupt_value = event['__interrupt__'][0].value
        display(Markdown(interrupt_value))
#%%
# 현재 상태를 직접 확인하고 수정
current_state = graph.get_state(thread)
print("현재 상태:", current_state.next)

# 강제로 다음 단계로 이동 시도
from langgraph.types import Command
async for event in graph.astream(
    Command(update={"completed_sections": [], "current_section": 0}), 
    thread, 
    stream_mode="updates"
):
    print(event)

#%%
final_state = graph.get_state(thread)
report = final_state.values.get('sections')
print(final_state)

#%%
# Display the final generated report
# Retrieve the completed report from the graph's state and format it for display

final_state = graph.get_state(thread)
report = final_state.values.get('final_report')
Markdown(report)