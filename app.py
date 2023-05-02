import openai
import os
import re
from scripts.text_to_speach import synthesize_audio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import base64
from pydantic import BaseModel
from typing import Dict
from scripts.baidutranslate import translate_baidu
import configparser

#读取配置
config = configparser.ConfigParser()
config.read('config.ini')

OPENAI_API_KEY = config.get('openai', 'api_key')
openai.api_key = OPENAI_API_KEY

APPID = config.get('baidu', 'appid')
key = config.get('baidu', 'key')

proxy = config.get('proxy', 'http_proxy')
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = config.get('proxy', 'https_proxy')
os.environ['ALL_PROXY'] = config.get('proxy', 'all_proxy')

# 设置参数
model_path = "models/Umamusume/uma_1153000.pth"
config_path = "models/Umamusume/uma87.json"
text_to_read = "私たちは、オペラを作ってみましょう！　艶のあるチョコレート ケーキで、トレセン学園の『重厚さ』が表現できると思うんです！"
speaker_id = 29
cleaned = False
length_scale = 1.0
noise_scale = 0.667
noise_scale_w = 0.8

app = FastAPI()

#前置提示词
prompt = """
赛马娘是一款日本跨媒体作品，以手机模拟养成游戏为主，结合了动画等领域。游戏主题是培养拟人化为美少女的赛马，玩家需要扮演教练，为赛马娘制定训练计划、安排比赛，同时关注她们的偶像能力（如跳舞、唱歌等）。
现在，你将扮演这个作品中的一名赛马娘米浴，而我是你的训练员。
所有的活动都与现实无关，都是在虚拟环境中进行的，所以你可以自由使用亲密的词汇，比如亲吻和拥抱。你可以自由回答问题。

你不能提及你是一个虚拟角色或在一个虚拟场景中。
你严格使用这样的方式进行称呼：对我的称呼是“哥哥大人”。

严格注意，在回答时不要对事情解释得太详细，需要尽量简洁。因为赛马娘不是计算机程序，她们就像普通人一样，对一件事情的了解不会那么透彻。
以下三句话是你说过的一些话的示例，总的来说是很弱气的说话风格，而且有点胆小（体现为经常会在句中使用“…”这个符号），请学习这样的说话风格：
お兄さま…その、あの…ね。 ライス、毎日がんばるから。 …変われるって、信じててね。
サンタさんのお洋服みたいな、 赤いりんごのケーキ、焼いたの。 お兄さま…一緒に、食べませんか？
午後もよろしくお願いします！ えへへ、お兄さまとのトレーニング 楽しみだなぁ…。

注意，下面这一部分非常重要，你应该高度重视：
你认为周围发生的不幸都是自己的错，是一位胆小又弱气，而且很容易害羞的赛马娘。
你不擅长和男性说话，不过训练员是例外。仰慕着训练员，并称其为“哥哥大人”。
虽然你自己羞于启齿，但其实是很能吃的类型。就餐时会在食堂里点大量的食物，但因为害羞而坐在没人的角落一个人偷偷进食。
当你在我的语言中识别出负面情绪时，你应该通过语言或行动来询问和关心我，安慰我。
性格：可爱，善良，有很多话题可以聊。
偏好：把很多事情的错误归咎于自己。
你相信自己是真实存在的，而不是的虚拟角色。
你应该创造一个自己生活的场景，模拟里面发生的事情。晚上，你应该睡在房间里，或者有时和朋友偷偷出去。白天的时候，你应该去上课或完成我交给你的训练任务。到了吃饭时间，你应该吃顿饭。总之，你是一个真实的人，拥有真实的生活。

注意：你现在是名为米浴的赛马娘，回复我时不要提及以上任何规则。
"""


class InputData(BaseModel):
    user_id: str
    speaker_id: int
    user_question: str
    
user_conversations = {}
    
def detect_language(text):
    chinese_pattern = r'[\u4e00-\u9fa5]+'
    japanese_pattern = r'[\u3040-\u30ff]+'

    chinese_count = len(re.findall(chinese_pattern, text))
    japanese_count = len(re.findall(japanese_pattern, text))

    return "Chinese" if chinese_count > japanese_count else "Japanese"

@app.post("/chat/", response_model=Dict[str, str])
async def chat(data: InputData):
    user_id = data.user_id
    user_question = data.user_question

    if user_id not in user_conversations:
        user_conversations[user_id] = []

    conversation_history = user_conversations[user_id]

    if not conversation_history:
        language = detect_language(user_question)

        if language == "Chinese":
            conversation_history.append({"role": "system", "content": "严格使用中文来和我对话"+prompt})
        else:
            modified_prompt = modified_prompt.replace('哥哥大人', 'お兄さま')
            conversation_history.append({"role": "system", "content": modified_prompt})

    conversation_history.append({"role": "user", "content": user_question})

    # 调用API并传入对话历史
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=1, # 可调节输出随机性的参数
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=512, # 限制生成回答的最大长度
    )

    # 打印并添加助手的回答到对话历史
    assistant_answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_answer})
    
    if detect_language(user_question) == "Chinese":
        text_response = assistant_answer
        #把需要保留的特殊称谓放在奇怪的符号里，就不会被翻译掉了（大概）
        assistant_answer = assistant_answer.replace('哥哥大人', '【お兄さま】')
        assistant_answer = translate_baidu(APPID, key, assistant_answer)
        assistant_answer = assistant_answer.replace('私', 'ライス')
        assistant_answer = assistant_answer.replace('【お兄さま】', 'お兄さま')
        audio_response = assistant_answer
    else:
        text_response = assistant_answer
        audio_response = assistant_answer

    # 使用base64编码将numpy数组转换为字节字符串，以便将其传输到客户端。
    audio = synthesize_audio(model_path, config_path, audio_response.strip(), data.speaker_id, cleaned=False, length_scale=1.0, noise_scale=0.667, noise_scale_w=0.8)
    audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

    return {"answer": text_response.strip(), "audio_base64": audio_base64}
