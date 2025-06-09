import uuid
import os
from IPython.display import Image, display, Markdown
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send

# open_deep_research의 기존 컴포넌트들 import
from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    ReportState,
)

from open_deep_research.configuration import Configuration

from open_deep_research.graph import (
    generate_report_plan,
    section_builder,
    gather_completed_sections,
    write_final_sections,
    compile_final_report,
    initiate_final_section_writing
)


def auto_approve_plan(state: ReportState, config):
    """자동으로 계획을 승인하고 섹션 작성을 시작하는 함수

    human_feedback를 대체하여 사용자 개입 없이 자동으로
    연구가 필요한 섹션들의 작성을 시작합니다.

    Args:
        state: 현재 그래프 상태 (섹션 정보 포함)
        config: 실행 설정

    Returns:
        Command: 병렬로 섹션 연구/작성을 시작하는 명령
    """
    topic = state["topic"]
    sections = state['sections']
    
    print(f"✅ 보고서 계획이 자동 승인되었습니다!")
    print(f"📝 총 {len(sections)}개 섹션 중 연구가 필요한 섹션: {len([s for s in sections if s.research])}개")
    
    # 연구가 필요한 섹션들을 병렬로 처리하도록 Send 명령 생성
    research_sections = [s for s in sections if s.research]
    
    if research_sections:
        # 연구가 필요한 섹션이 있으면 병렬 처리
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic,
                "section": s,
                "search_iterations": 0
            })
            for s in research_sections
        ])
    else:
        # 연구가 필요한 섹션이 없으면 바로 gather_completed_sections로
        print("⚠️ 연구가 필요한 섹션이 없습니다. 바로 최종 섹션 작성으로 이동합니다.")
        return Command(goto="gather_completed_sections")


# 커스텀 그래프 생성
print("🔧 커스텀 연구 에이전트 그래프를 생성하는 중...")

custom_builder = StateGraph(
    ReportState,
    input=ReportStateInput,
    output=ReportStateOutput,
    config_schema=Configuration
)

# 노드 추가 (human_feedback 대신 auto_approve_plan 사용)
custom_builder.add_node("generate_report_plan", generate_report_plan)
custom_builder.add_node("auto_approve", auto_approve_plan)  # 핵심 변경점!
custom_builder.add_node("build_section_with_web_research", section_builder.compile())
custom_builder.add_node("gather_completed_sections", gather_completed_sections)
custom_builder.add_node("write_final_sections", write_final_sections)
custom_builder.add_node("compile_final_report", compile_final_report)

# 엣지 연결 (워크플로우 정의)
custom_builder.add_edge(START, "generate_report_plan")
custom_builder.add_edge("generate_report_plan", "auto_approve")  # human_feedback → auto_approve
custom_builder.add_edge("build_section_with_web_research", "gather_completed_sections")
custom_builder.add_conditional_edges(
    "gather_completed_sections",
    initiate_final_section_writing,
    ["write_final_sections"]
)
custom_builder.add_edge("write_final_sections", "compile_final_report")
custom_builder.add_edge("compile_final_report", END)

# 메모리 체크포인터와 함께 그래프 컴파일
memory = MemorySaver()
custom_graph = custom_builder.compile(checkpointer=memory)

print("✅ 커스텀 그래프 생성 완료!")

# 그래프 시각화 (선택적)
try:
    print("📊 그래프 구조를 시각화하는 중...")
    display(Image(custom_graph.get_graph(xray=1).draw_mermaid_png()))
except Exception as e:
    print(f"시각화 건너뛰기: {e}")

# 설정 예제
REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections
   - Provide a concise summary of the report"""


# 사용 예제 함수
async def run_auto_research(topic, search_api="tavily", max_search_depth=1):
    """자동 연구 에이전트를 실행하는 함수

    Args:
        topic: 연구할 주제
        search_api: 사용할 검색 API (기본값: "tavily")
        max_search_depth: 최대 검색 깊이 (기본값: 1)

    Returns:
        완성된 보고서 문자열
    """
    
    # 스레드 설정
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": search_api,
            "planner_provider": "groq",
            "planner_model": "deepseek-r1-distill-llama-70b",
            "writer_provider": "groq",
            "writer_model": "llama-3.3-70b-versatile",
            "report_structure": REPORT_STRUCTURE,
            "max_search_depth": max_search_depth,
        }
    }
    
    print(f"🚀 '{topic}' 주제로 자동 연구를 시작합니다...")
    
    # 그래프 실행 (자동으로 끝까지 실행됨)
    final_state = None
    async for event in custom_graph.astream({"topic": topic}, thread, stream_mode="updates"):
        # 진행 상황 출력 (선택적)
        for node_name, node_data in event.items():
            if node_name != "__interrupt__":
                print(f"📋 {node_name} 노드 완료")
        
        # 최종 상태 저장
        if "__interrupt__" not in event:
            final_state = custom_graph.get_state(thread)
    
    # 최종 보고서 반환
    if final_state and final_state.values.get('final_report'):
        return final_state.values['final_report']
    else:
        return "보고서 생성에 실패했습니다."


# 사용법 예제
print("\n🎯 사용법:")
print("""
# 비동기 실행 예제:
topic = "Overview of Model Context Protocol (MCP)"
report = await run_auto_research(topic)
print(report)

# 또는 직접 실행:
thread = {"configurable": {"thread_id": str(uuid.uuid4()), ...}}
async for event in custom_graph.astream({"topic": topic}, thread):
    pass
final_report = custom_graph.get_state(thread).values.get('final_report')
""")

print("\n🎉 자동 승인 연구 에이전트가 준비되었습니다!")
print("이제 사용자 개입 없이 자동으로 보고서를 생성할 수 있습니다.")