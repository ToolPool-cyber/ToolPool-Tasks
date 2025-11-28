# -*- coding: utf-8 -*-
import json
import copy
from typing import Optional, Dict, Any, List, TypedDict, AsyncGenerator
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import asyncio
import uuid
import requests
import aiohttp
import logging
from fastapi.middleware.cors import CORSMiddleware
from ragflow_sdk import RAGFlow

# 设置日志记录
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务器地址配置
CTC_URL = "http://10.103.238.156:9451/ctc_server"  # 题目匹配服务
DISCUSSION_URL = "http://10.103.238.156:9452/discussion"  # 问答服务
LLM_URL = "http://10.103.238.156:9453/chat"  # 大模型服务 (现在是流式)
OCR_URL = "http://10.103.238.156:9454/ocr_inference"  # OCR 服务
H5_URL = "http://10.103.238.156:9457/H5_server"  # H5服务
HardWare_URL = "http://10.103.238.156:9458/HardWare_server"  # 硬件实验服务
HardWareGetArgs_URL = "http://10.103.238.156:9455/GetArgs"  # 自然语言转硬件实验设置参数
COURSE_EXPERIMENT_URL = "http://10.103.238.156:8001/find_experiments"  # 1.py提供的课程实验查询服务

# SDK集成: RagFlow SDK 配置
RAGFLOW_SDK_BASE_URL = "http://10.103.238.156:9380"
RAGFLOW_API_KEY = "ragflow-I1ZTM2N2I2MjM2ZjExZjBhODRiZGEyOD"
RAGFLOW_TARGET_DATASET_NAME = "通原"

target_dataset_ids_for_ragflow: List[str] = []
ragflow_client: Optional[RAGFlow] = None


@app.on_event("startup")
async def startup_event():
    global ragflow_client, target_dataset_ids_for_ragflow
    logging.info("Application startup: Initializing RAGFlow client...")
    if not RAGFLOW_API_KEY or RAGFLOW_API_KEY == "YOUR_RAGFLOW_API_KEY_HERE":
        logging.error("RAGFlow API Key 未有效配置，SDK客户端初始化跳过。")
        return

    try:
        ragflow_client = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_SDK_BASE_URL)
        logging.info(f"RAGFlow SDK客户端实例已创建，连接到: {RAGFLOW_SDK_BASE_URL}")

        logging.info(f"尝试获取数据集 '{RAGFLOW_TARGET_DATASET_NAME}' 的ID...")
        datasets_found = await asyncio.to_thread(
            ragflow_client.list_datasets,
            name=RAGFLOW_TARGET_DATASET_NAME
        )

        if datasets_found and hasattr(datasets_found[0], 'id'):
            target_dataset_ids_for_ragflow = [str(datasets_found[0].id)]
            logging.info(f"成功获取到数据集 '{RAGFLOW_TARGET_DATASET_NAME}' 的ID: {target_dataset_ids_for_ragflow}")
        else:
            logging.error(f"未能找到名为 '{RAGFLOW_TARGET_DATASET_NAME}' 的数据集或返回对象结构不符合预期。")
            target_dataset_ids_for_ragflow = []

    except Exception as e:
        logging.error(f"初始化RAGFlow SDK客户端或获取数据集ID时发生错误: {e}", exc_info=True)
        ragflow_client = None
        target_dataset_ids_for_ragflow = []


templates = Jinja2Templates(directory="templates")

# 临时文件保存目录
TEMP_FILE_DIR = "temp_files"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)


# 数据模型定义
class Attachment(BaseModel): id: str; type: str


class ChatMessage(BaseModel): chat_id: int; content: str; attachments: List[Attachment] = []


class ChatMessage_H5(BaseModel): chat_id: int; content: str; H5_List: List; attachments: List[Attachment] = []


# 业务类型模板
business_types_template = {
    '问题内容': '', '通信原理题目库': 0, '通信原理课堂教师答疑记录': 0,
    '大模型概念回答': 0, '通信原理课堂交互式案例': 0, '通信原理课堂在线硬件实验': 0,
    '其他课程实验案例': 0
}


# 显示前端页面
@app.get("/", response_class=HTMLResponse)
async def get_frontend_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 保存上传的图片文件到本地
async def save_image_locally(image_file: UploadFile) -> str:
    try:
        file_ext = image_file.filename.split('.')[-1]
        unique_filename = str(uuid.uuid4()) + "." + file_ext
        temp_file_path = os.path.join(TEMP_FILE_DIR, unique_filename)
        with open(temp_file_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)
        logging.info(f"图片已保存到临时文件: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logging.error(f"保存图片时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"保存图片时发生错误: {str(e)}")


# OCR识别相关函数
async def get_ocr_text_async_wrapper(file_path: str) -> str:
    return await asyncio.to_thread(get_ocr_text_sync, file_path)


def get_ocr_text_sync(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(OCR_URL, files={"file": file})
        response.raise_for_status()
        ocr_result = response.json()
        logging.info(f"OCR完成 ,返回结果: {ocr_result}")
        return ocr_result["result"]
    except Exception as e:
        logging.error(f"OCR处理发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR处理发生错误: {str(e)}")


# -------------------- 核心工具：处理流式 LLM 请求 --------------------

async def get_llm_response_full(payload: dict) -> str:
    """
    工具函数：调用流式LLM接口，但将结果拼接成完整字符串返回。
    用于意图识别（需要完整的JSON结构，不能流式解析）。
    """
    full_content = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    logging.error(f"LLM request failed with status {response.status}")
                    return ""

                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                delta = data_json["choices"][0]["delta"].get("content", "")
                                full_content += delta
                        except Exception:
                            pass
        return full_content
    except Exception as e:
        logging.error(f"Error getting full LLM response: {e}")
        return ""


# -------------------- 业务逻辑处理函数 --------------------

async def auto_select_business_type(question: str, model_type: str) -> dict:
    """意图识别：调用LLM获取JSON配置，必须等待完全返回"""
    prompt = (
        f"{question}\n根据上述的问题选择最合适的业务类型，并按格式返回对应的数字：\n返回内容有且只有如下格式内容，不要添加任何多余文字、字符：\n"
        "{'问题内容':'xxx','通信原理题目库': 1, '通信原理课堂教师答疑记录': 1, '大模型概念回答': 1,'通信原理课堂交互式案例': 1,'通信原理课堂在线硬件实验': 1,'其他课程实验案例': 1}\n"
        "我们有多个可用的业务类型：\n1. 通信原理题目库（里面有大量的通信原理相关习题）\n2. 通信原理课堂教师答疑记录\n3. 大模型概念回答（用于概要介绍通信原理相关概念的大模型）\n"
        "4. 通信原理课堂交互式案例(可以在线进行相关的软件仿真实验)\n5. 通信原理课堂在线硬件实验(可以在线进行相关的硬件仿真实验)\n"
        "6. 其他课程实验案例(非通信原理的其他课程，如电子技术基础、数字电路等的相关实验案例，只要问题不属于通信原理，均优先标记此类)\n"
        "请分析问题，并根据问题的内容判断哪些业务类型相关。\n对于每个业务类型，如果问题相关，请返回 1；如果不相关，请返回 0。\n请以字符串的格式返回，字典中每个业务类型的值为 1 或 0，表示该业务类型是否与问题相关。\n"
    )

    payload = {"message": prompt, "model": model_type}
    logging.info("--- Intent Recognition Start ---")

    full_text = await get_llm_response_full(payload)
    logging.info(f"Intent Raw Response: {full_text}")

    try:
        clean_text = full_text.replace("```json", "").replace("```", "").strip()
        str_data = clean_text.replace("'", '"')
        content = json.loads(str_data)

        if '问题内容' not in content or not content['问题内容']:
            content['问题内容'] = question
        return content
    except Exception as e:
        logging.error(f"Intent JSON parse error: {e}")
        return {'问题内容': question, '通信原理题目库': 1, '大模型概念回答': 1}


# -------------------- 独立的后台任务函数 --------------------

async def fetch_other_course_experiments(question: str, model_type: str) -> dict:
    payload = {"question": question, "model_type": model_type}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(COURSE_EXPERIMENT_URL, json=payload) as response:
                if response.status != 200:
                    return {"content": "查询实验案例时服务调用失败", "list": []}

                res_json = await response.json()
                course_name = res_json.get("course_name", "")

                if course_name == "通信原理":
                    return {"content": "该问题属于通信原理课程，相关实验请参考通信原理实验内容", "list": []}

                if not res_json.get("success", False):
                    return {"content": res_json.get("message", "未找到相关实验案例"), "list": []}

                experiments = res_json.get("experiments", [])
                if not experiments:
                    return {"content": f"在'{course_name}'课程中未找到相关实验案例", "list": []}

                content_str = f"根据您的提问，在'{course_name}'课程中为您找到以下实验案例：\n"
                for i, exp in enumerate(experiments, 1):
                    content_str += f"{i}. {exp.get('key_word', '')}\n"

                return {
                    "content": content_str,
                    "list": [{"实验案例": exp.get("key_word", ""), "链接": exp.get("url", "")} for exp in experiments]
                }
    except Exception as e:
        logging.error(f"Other course experiment error: {e}")
        return {"content": "查询实验案例时发生异常", "list": []}


def fetch_question_bank_sync(question: str) -> dict:
    try:
        response = requests.post(CTC_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            return {"content": result.get("content", ""), "list": result.get("list", [])}
    except Exception:
        pass
    return {"content": "通信原理题库请求失败", "list": []}


def fetch_qa_record_sync(question: str) -> dict:
    try:
        response = requests.post(DISCUSSION_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            return {"content": result.get("content", ""), "list": result.get("list", [])}
    except Exception:
        pass
    return {"content": "教师问答记录请求失败", "list": []}


def fetch_interactive_stories_sync(question: str) -> dict:
    try:
        response = requests.post(H5_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            return {"content": result.get("content", ""), "list": result.get("H5_List", [])}
    except Exception:
        pass
    return {"content": "交互式案例请求失败", "list": []}


def fetch_hardware_experiments_sync(question: str) -> dict:
    try:
        response = requests.post(HardWare_URL, json={"message": question})
        args_response = requests.post(HardWareGetArgs_URL, json={"message": question})
        if response.status_code == 200 and args_response.status_code == 200:
            result = response.json()
            args_dict = args_response.json()
            return {
                "content": result.get("content", ""),
                "list": result.get("H5_List", []),
                "HardWareArgs": args_dict.get("HardWareArgs", {})
            }
    except Exception:
        pass
    return {"content": "硬件实验请求失败", "list": [], "HardWareArgs": {}}


# -------------------- 主流式生成器 --------------------

async def stream_concept_answer_generator(question: str, model: str, rag_chunks: str) -> AsyncGenerator[str, None]:
    """负责调用大模型并逐字yield token (概念回答)"""
    if rag_chunks:
        prompt = (f"请结合以下知识库信息（如果与问题相关）以及你已有的知识，用较少的文字在通信领域内回答问题。\n"
                  f"{rag_chunks}\n\n用户的问题是：{question}")
    else:
        prompt = (f"用较少的文字来回答下面问题中，回答范围限定在通信领域, 问题：{question}")

    payload = {"message": prompt, "model": model}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    yield f"Error: LLM service returned {response.status}"
                    return

                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                content = data_json["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            elif "type" in data_json and data_json["type"] == "error":
                                yield f"Error: {data_json.get('content')}"
                        except Exception:
                            continue
    except Exception as e:
        yield f"Error during streaming: {str(e)}"


async def stream_next_question_generator(question: str, answer: str, model_type: str) -> AsyncGenerator[str, None]:
    """
    负责调用大模型并逐字yield token (下一步问题预测)
    """
    prompt = (
        f"用户提出的问题：{question}\n大模型返回的答案：{answer}\n"
        "请预测下一步用户可能会问的n个问题（最多4个），格式如下：\n"
        "{'0':'question_1','1':'question_2', ...}\n"
        "请只返回JSON字符串，不要包含多余文字。\n"
    )
    payload = {"message": prompt, "model": model_type}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    yield ""  # 错误时静默或返回空
                    return

                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                content = data_json["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                        except Exception:
                            continue
    except Exception as e:
        logging.error(f"Error streaming next questions: {e}")
        yield ""


async def distribute_tasks_generator(
        user_config: dict,
        user_question: str,
        ocr_text: Optional[str] = None
):
    """
    调度中心生成器：
    1. 识别意图 -> yield 意图配置
    2. 启动后台任务 (数据库查询等)
    3. 流式传输大模型回答 -> yield tokens
    4. 等待后台任务完成 -> yield 任务结果
    5. 流式传输下一步问题 -> yield tokens (next_question_token)
    """

    # 1. 构造用于意图识别的组合问题
    combined_question = user_question
    if ocr_text:
        combined_question = (
            f"用户的文字提问: 在通信领域里：{user_question}\n用户的图片提问里的文字: {ocr_text}\n"
            f"请结合上述两个信息，生成一个连贯的问题。"
        )

    # 2. 意图识别 (需要等待完整结果)
    business_types = copy.deepcopy(business_types_template)
    try:
        identified_config = await auto_select_business_type(
            combined_question, model_type=user_config.get('model', 'DeepSeek')
        )
        business_types.update(identified_config)
    except Exception as e:
        logging.error(f"Intent recognition failed: {e}")
        business_types['问题内容'] = combined_question
        business_types['大模型概念回答'] = 1  # 降级默认开启

    business_types["大模型概念回答"] = 1  # 强制开启
    final_question = business_types.get('问题内容', combined_question)

    # ---> 推送给前端：意图配置
    yield f"data: {json.dumps({'type': 'config', 'data': business_types}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0.01)

    # 3. 启动并行任务 (不等待)
    background_tasks = {}

    if business_types.get("通信原理题目库", 0) == 1:
        background_tasks["通信原理题目库"] = asyncio.create_task(
            asyncio.to_thread(fetch_question_bank_sync, final_question)
        )
    if business_types.get("通信原理课堂教师答疑记录", 0) == 1:
        background_tasks["通信原理课堂教师答疑记录"] = asyncio.create_task(
            asyncio.to_thread(fetch_qa_record_sync, final_question)
        )
    if business_types.get("通信原理课堂交互式案例", 0) == 1:
        background_tasks["通信原理课堂交互式案例"] = asyncio.create_task(
            asyncio.to_thread(fetch_interactive_stories_sync, final_question)
        )
    if business_types.get("通信原理课堂在线硬件实验", 0) == 1:
        background_tasks["通信原理课堂在线硬件实验"] = asyncio.create_task(
            asyncio.to_thread(fetch_hardware_experiments_sync, final_question)
        )
    if business_types.get("其他课程实验案例", 0) == 1:
        background_tasks["其他课程实验案例"] = asyncio.create_task(
            fetch_other_course_experiments(final_question, user_config.get('model', 'DeepSeek'))
        )

    # 4. 准备 RAG 数据 (SDK检索)
    rag_chunks_text = ""
    if ragflow_client and target_dataset_ids_for_ragflow:
        try:
            # === 修改开始：使用原生 requests 替代 SDK 调用，规避版本不兼容问题 ===
            logging.info(f"正在手动调用 RAG 检索接口...")

            def manual_retrieve_task():
                # 手动构造请求地址和数据，确保字段名是 dataset_ids
                url = f"{RAGFLOW_SDK_BASE_URL}/api/v1/retrieval"
                headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
                payload = {
                    "dataset_ids": target_dataset_ids_for_ragflow,  # 明确指定字段名
                    "question": final_question,
                    "similarity_threshold": 0.3,
                    "top_k": 3  # 这里控制返回数量
                }

                # 发送请求
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()

            # 在线程中执行，防止阻塞主程序
            rag_response = await asyncio.to_thread(manual_retrieve_task)

            # 解析返回结果 (RAGFlow API 通常返回 {"data": [...]})
            retrieved_items = []
            if isinstance(rag_response, dict) and 'data' in rag_response:
                retrieved_items = rag_response['data']
            elif isinstance(rag_response, dict) and 'code' in rag_response and rag_response['code'] != 0:
                logging.error(f"RAG API 返回错误: {rag_response}")

            logging.info(f"RAG 检索成功，获取到 {len(retrieved_items)} 条数据")
            # === 修改结束 ===

            if retrieved_items:
                chunks_builder = ["\n\n以下是知识库中检索到的相关参考信息："]
                for i, item in enumerate(retrieved_items):
                    # === 修复：增加对纯字符串返回值的支持 ===
                    if isinstance(item, str):
                        c = item
                    # 兼容字典格式 (API返回)
                    elif isinstance(item, dict):
                        c = item.get('content_with_weight', item.get('content', ''))
                    # 兼容对象格式 (SDK返回)
                    else:
                        c = getattr(item, 'content', str(item))

                    if c:
                        chunks_builder.append(f"参考片段 {i + 1}: {c}")
                rag_chunks_text = "\n".join(chunks_builder)

        except Exception as e:
            logging.error(f"RAG retrieve error: {e}")

    # 5. 流式输出大模型回答 (Token by Token)
    full_answer_accumulator = ""

    async for token in stream_concept_answer_generator(final_question, user_config.get('model', 'DeepSeek'),
                                                       rag_chunks_text):
        full_answer_accumulator += token
        # ---> 推送给前端：Token
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)

    # 6. 等待并推送后台任务结果
    for key, task in background_tasks.items():
        try:
            result_data = await task
            # ---> 推送给前端：板块数据
            payload = {"type": "section", "key": key, "data": result_data}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            logging.error(f"Task {key} failed: {e}")
            yield f"data: {json.dumps({'type': 'section', 'key': key, 'data': {'content': '获取失败'}}, ensure_ascii=False)}\n\n"

    # 7. 流式输出下一步问题预测 (New Streaming Implementation)
    if full_answer_accumulator and not full_answer_accumulator.startswith("Error"):
        # 发送开始信号（可选，视前端是否需要清理区域）
        yield f"data: {json.dumps({'type': 'next_questions_start'}, ensure_ascii=False)}\n\n"

        async for next_q_token in stream_next_question_generator(
                final_question,
                full_answer_accumulator,
                user_config.get('model', 'DeepSeek')
        ):
            # ---> 推送给前端：Next Question Token
            # 前端需监听 'next_question_token' 事件，并拼接 content 字符串
            # 注意：由于Prompt要求返回JSON，这里流式传输的是JSON字符串
            yield f"data: {json.dumps({'type': 'next_question_token', 'content': next_q_token}, ensure_ascii=False)}\n\n"

    # 结束标记
    yield "data: [DONE]\n\n"


@app.post("/tasks_distribute")
async def tasks_distribute(
        config: str = Form(...), entrance: str = Form(...), locale: str = Form(...),
        image: Optional[UploadFile] = File(None),
):
    try:
        # 解析配置
        message_from_form_str = config
        str_data = message_from_form_str.replace("'", '"')
        message_config = json.loads(str_data)

        # 处理图片 OCR
        ocr_text = None
        if image is not None:
            temp_file_path = await save_image_locally(image)
            ocr_text = await get_ocr_text_async_wrapper(temp_file_path)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        user_original_question = message_config.get('content', '')

        # 返回 StreamingResponse，启用 SSE
        return StreamingResponse(
            distribute_tasks_generator(message_config, user_original_question, ocr_text),
            media_type="text/event-stream"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Config JSON Error: {str(e)}")
    except Exception as e:
        logging.error(f"Main error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=9450)