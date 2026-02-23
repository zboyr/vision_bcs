#!/usr/bin/env python3
"""
setup_limesurvey.py
通过 LimeSurvey RemoteControl 2 API 创建 BCS 评分问卷。

策略：
1. 构建 LSS XML（含英文+中文双语、条件逻辑）
2. 通过 import_survey API 一次性导入
3. 图片通过外部 HTTPS URL 引用（image_urls.json）

特性：
- 英文 + 简体中文双语
- 第三个问题（第二 BCS 评分）仅在置信度选 B 或 C 时显示
- 50 张猫图片，每张 3 个问题

使用方法:
    python3 setup_limesurvey.py

LimeSurvey 服务器: shawarmai.com/bcs
管理员: admin / fh4829hd0h.
"""

import base64
import csv
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# LimeSurvey 配置
LS_URL = "https://shawarmai.com/bcs/index.php/admin/remotecontrol"
LS_USER = "admin"
LS_PASS = "fh4829hd0h."

# ── 英文文本 ──────────────────────────────────────────────
SURVEY_TITLE_EN = "Feline BCS Visual Assessment Quiz"
SURVEY_DESC_EN = (
    "Please assess the Body Condition Score (BCS) of each cat based solely "
    "on visual assessment from the photograph. BCS uses a 9-point scale "
    "(1 = emaciated, 5 = ideal, 9 = obese)."
)
SURVEY_WELCOME_EN = """<h2>Feline Body Condition Score (BCS) Visual Assessment</h2>
<p>Thank you for participating in this study on feline body condition scoring.</p>
<p><strong>Instructions:</strong></p>
<ul>
<li>Review each image of a cat carefully</li>
<li>Determine the BCS for the cat based on visual assessment only</li>
<li>Select a BCS score from 1-9 (whole numbers only)</li>
<li>Indicate your confidence level:
  <ul>
    <li><strong>A</strong>: You are &gt;90% confident in your single BCS score</li>
    <li><strong>B</strong>: Two adjacent scores are equally likely</li>
    <li><strong>C</strong>: Two scores are possible but you lean toward one (~75:25)</li>
  </ul>
</li>
<li>If you select confidence B or C, please also provide your second BCS score</li>
</ul>
<p>There are no right or wrong answers. Please answer as accurately as possible.</p>"""

SURVEY_END_EN = """<h3>Thank you for participating in our BCS Visual Assessment Quiz!</h3>
<p>Your responses have been recorded. If you have any questions, please contact
Dr. Xu Wang (xzw0070@auburn.edu) or Dr. Emily Graff (ecg0001@auburn.edu).</p>"""

# ── 中文文本 ──────────────────────────────────────────────
SURVEY_TITLE_ZH = "猫体况评分 (BCS) 视觉评估问卷"
SURVEY_DESC_ZH = (
    "请仅根据照片对每只猫的体况评分 (BCS) 进行视觉评估。"
    "BCS 采用 9 分制（1 = 极度消瘦，5 = 理想体态，9 = 肥胖）。"
)
SURVEY_WELCOME_ZH = """<h2>猫体况评分 (BCS) 视觉评估</h2>
<p>感谢您参与本次猫体况评分研究。</p>
<p><strong>说明：</strong></p>
<ul>
<li>仔细观察每张猫的照片</li>
<li>仅根据视觉评估判断该猫的 BCS</li>
<li>选择 1-9 的整数 BCS 评分</li>
<li>选择您的置信度：
  <ul>
    <li><strong>A</strong>：您对自己的单一 BCS 评分有 &gt;90% 的把握</li>
    <li><strong>B</strong>：两个相邻评分同样可能</li>
    <li><strong>C</strong>：两个评分都有可能，但您倾向于其中一个（约 75:25）</li>
  </ul>
</li>
<li>如果选择置信度 B 或 C，请同时提供您的第二个 BCS 评分</li>
</ul>
<p>没有标准答案，请尽可能准确地作答。</p>"""

SURVEY_END_ZH = """<h3>感谢您参与猫 BCS 视觉评估问卷！</h3>
<p>您的回答已记录。如有任何问题，请联系
Dr. Xu Wang (xzw0070@auburn.edu) 或 Dr. Emily Graff (ecg0001@auburn.edu)。</p>"""


# ── API 客户端 ────────────────────────────────────────────
class LimeSurveyAPI:
    """LimeSurvey RemoteControl 2 API 客户端。"""

    def __init__(self, url, username, password):
        self.url = url
        self.username = username
        self.password = password
        self.session_key = None
        self._request_id = 0

    def _call(self, method, params):
        self._request_id += 1
        payload = {"method": method, "params": params, "id": self._request_id}
        headers = {"content-type": "application/json", "connection": "Keep-Alive"}
        try:
            r = requests.post(self.url, json=payload, headers=headers,
                              timeout=120, verify=True)
            r.raise_for_status()
            result = r.json()
            if "error" in result and result["error"] is not None:
                raise Exception(f"API 错误: {result['error']}")
            return result.get("result")
        except requests.exceptions.SSLError:
            r = requests.post(self.url, json=payload, headers=headers,
                              timeout=120, verify=False)
            r.raise_for_status()
            result = r.json()
            if "error" in result and result["error"] is not None:
                raise Exception(f"API 错误: {result['error']}")
            return result.get("result")

    def connect(self):
        result = self._call("get_session_key", [self.username, self.password])
        if isinstance(result, dict) and "status" in result:
            raise Exception(f"登录失败: {result['status']}")
        self.session_key = result
        print(f"已连接到 LimeSurvey (session: {self.session_key[:8]}...)")

    def disconnect(self):
        if self.session_key:
            self._call("release_session_key", [self.session_key])
            self.session_key = None
            print("已断开连接")

    def import_survey(self, import_data, import_type="lss"):
        return self._call("import_survey",
                          [self.session_key, import_data, import_type])

    def set_survey_properties(self, sid, props):
        return self._call("set_survey_properties",
                          [self.session_key, sid, props])

    def activate_survey(self, sid):
        return self._call("activate_survey", [self.session_key, sid])

    def list_surveys(self):
        return self._call("list_surveys", [self.session_key])

    def delete_survey(self, sid):
        return self._call("delete_survey", [self.session_key, sid])

    def get_question_properties(self, qid, settings):
        return self._call("get_question_properties",
                          [self.session_key, qid, settings])


# ── 工具函数 ──────────────────────────────────────────────
def load_dataset():
    csv_path = os.path.join(BASE_DIR, "dataset.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_image_urls():
    p = os.path.join(BASE_DIR, "image_urls.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}


# ── LSS XML 构建 ─────────────────────────────────────────
def build_lss_xml(records):
    """
    构建完整的 LSS XML，包含：
    - 英文 (en) + 简体中文 (zh-Hans) 双语
    - 条件逻辑：第三个问题仅在置信度选 B/C 时显示
    """
    image_urls = load_image_urls()
    sid = 100001

    groups_rows = ""
    questions_rows = ""
    answers_rows = ""

    qid = 1
    for i, record in enumerate(records):
        cat_id = int(record["image_id"])
        gid = i + 1

        img_url = image_urls.get(str(cat_id), "")
        if img_url:
            img_en = (
                f'<div style="text-align:center;margin:20px 0;">'
                f'<img src="{img_url}" alt="Cat #{cat_id}" '
                f'style="max-width:600px;max-height:500px;border:1px solid #ccc;'
                f'border-radius:8px;" />'
                f'</div>'
            )
            img_zh = img_en  # 图片两种语言共用
        else:
            img_en = f'<p style="color:red;">Image missing: Cat #{cat_id}</p>'
            img_zh = f'<p style="color:red;">图片缺失：猫 #{cat_id}</p>'

        # ── 问题组（双语）──
        for lang, name in [("en", f"Cat #{cat_id}"), ("zh-Hans", f"猫 #{cat_id}")]:
            groups_rows += f"""
      <row>
        <gid><![CDATA[{gid}]]></gid>
        <sid><![CDATA[{sid}]]></sid>
        <group_name><![CDATA[{name}]]></group_name>
        <group_order><![CDATA[{i}]]></group_order>
        <description><![CDATA[]]></description>
        <language><![CDATA[{lang}]]></language>
        <randomization_group><![CDATA[]]></randomization_group>
        <grelevance><![CDATA[]]></grelevance>
      </row>"""

        # ── Q1: BCS 评分（双语）──
        bcs_qid = qid
        bcs_title = f"bcs{cat_id:02d}"
        bcs_en = (f'{img_en}<p>Based on the photograph above, what BCS (Body '
                  f'Condition Score) would you assign to <strong>Cat #{cat_id}'
                  f'</strong>?</p>')
        bcs_zh = (f'{img_zh}<p>根据上方照片，您会给<strong>猫 #{cat_id}'
                  f'</strong>打多少 BCS（体况评分）？</p>')

        for lang, text in [("en", bcs_en), ("zh-Hans", bcs_zh)]:
            questions_rows += f"""
      <row>
        <qid><![CDATA[{bcs_qid}]]></qid>
        <parent_qid><![CDATA[0]]></parent_qid>
        <sid><![CDATA[{sid}]]></sid>
        <gid><![CDATA[{gid}]]></gid>
        <type><![CDATA[L]]></type>
        <title><![CDATA[{bcs_title}]]></title>
        <question><![CDATA[{text}]]></question>
        <help><![CDATA[]]></help>
        <preg><![CDATA[]]></preg>
        <other><![CDATA[N]]></other>
        <mandatory><![CDATA[Y]]></mandatory>
        <encrypted><![CDATA[N]]></encrypted>
        <question_order><![CDATA[0]]></question_order>
        <scale_id><![CDATA[0]]></scale_id>
        <same_default><![CDATA[0]]></same_default>
        <relevance><![CDATA[1]]></relevance>
        <question_theme_name><![CDATA[listradio]]></question_theme_name>
        <modulename/>
        <same_script><![CDATA[0]]></same_script>
        <language><![CDATA[{lang}]]></language>
      </row>"""

        # BCS 答案 1-9（双语，数字相同）
        for score in range(1, 10):
            for lang in ("en", "zh-Hans"):
                answers_rows += f"""
      <row>
        <qid><![CDATA[{bcs_qid}]]></qid>
        <code><![CDATA[{score}]]></code>
        <answer><![CDATA[{score}]]></answer>
        <sortorder><![CDATA[{score}]]></sortorder>
        <assessment_value><![CDATA[{score}]]></assessment_value>
        <language><![CDATA[{lang}]]></language>
        <scale_id><![CDATA[0]]></scale_id>
      </row>"""
        qid += 1

        # ── Q2: 置信度（双语）──
        conf_qid = qid
        conf_title = f"conf{cat_id:02d}"
        conf_en = "How confident are you in your BCS assessment?"
        conf_zh = "您对自己的 BCS 评分有多大把握？"

        for lang, text in [("en", conf_en), ("zh-Hans", conf_zh)]:
            questions_rows += f"""
      <row>
        <qid><![CDATA[{conf_qid}]]></qid>
        <parent_qid><![CDATA[0]]></parent_qid>
        <sid><![CDATA[{sid}]]></sid>
        <gid><![CDATA[{gid}]]></gid>
        <type><![CDATA[L]]></type>
        <title><![CDATA[{conf_title}]]></title>
        <question><![CDATA[{text}]]></question>
        <help><![CDATA[]]></help>
        <preg><![CDATA[]]></preg>
        <other><![CDATA[N]]></other>
        <mandatory><![CDATA[Y]]></mandatory>
        <encrypted><![CDATA[N]]></encrypted>
        <question_order><![CDATA[1]]></question_order>
        <scale_id><![CDATA[0]]></scale_id>
        <same_default><![CDATA[0]]></same_default>
        <relevance><![CDATA[1]]></relevance>
        <question_theme_name><![CDATA[listradio]]></question_theme_name>
        <modulename/>
        <same_script><![CDATA[0]]></same_script>
        <language><![CDATA[{lang}]]></language>
      </row>"""

        # 置信度答案（双语）
        conf_opts = [
            ("A",
             "A - I am &gt;90% confident in my single BCS score",
             "A - 我对自己的单一 BCS 评分有 &gt;90% 的把握"),
            ("B",
             "B - Two adjacent scores are equally likely",
             "B - 两个相邻评分同样可能"),
            ("C",
             "C - Two scores are possible but I lean toward one (~75:25)",
             "C - 两个评分都有可能，但我倾向于其中一个（约 75:25）"),
        ]
        for ci, (code, text_en, text_zh) in enumerate(conf_opts):
            for lang, text in [("en", text_en), ("zh-Hans", text_zh)]:
                answers_rows += f"""
      <row>
        <qid><![CDATA[{conf_qid}]]></qid>
        <code><![CDATA[{code}]]></code>
        <answer><![CDATA[{text}]]></answer>
        <sortorder><![CDATA[{ci + 1}]]></sortorder>
        <assessment_value><![CDATA[0]]></assessment_value>
        <language><![CDATA[{lang}]]></language>
        <scale_id><![CDATA[0]]></scale_id>
      </row>"""
        qid += 1

        # ── Q3: 第二 BCS（条件显示，双语）──
        bcs2_qid = qid
        bcs2_title = f"bcstwo{cat_id:02d}"
        # 条件：仅当同组置信度问题选了 B 或 C 时才显示
        relevance_expr = (f'{conf_title}.NAOK == "B" || '
                          f'{conf_title}.NAOK == "C"')
        bcs2_en = ("Please provide your second BCS score.")
        bcs2_zh = ("请提供您的第二个 BCS 评分。")

        for lang, text in [("en", bcs2_en), ("zh-Hans", bcs2_zh)]:
            questions_rows += f"""
      <row>
        <qid><![CDATA[{bcs2_qid}]]></qid>
        <parent_qid><![CDATA[0]]></parent_qid>
        <sid><![CDATA[{sid}]]></sid>
        <gid><![CDATA[{gid}]]></gid>
        <type><![CDATA[L]]></type>
        <title><![CDATA[{bcs2_title}]]></title>
        <question><![CDATA[{text}]]></question>
        <help><![CDATA[]]></help>
        <preg><![CDATA[]]></preg>
        <other><![CDATA[N]]></other>
        <mandatory><![CDATA[Y]]></mandatory>
        <encrypted><![CDATA[N]]></encrypted>
        <question_order><![CDATA[2]]></question_order>
        <scale_id><![CDATA[0]]></scale_id>
        <same_default><![CDATA[0]]></same_default>
        <relevance><![CDATA[{relevance_expr}]]></relevance>
        <question_theme_name><![CDATA[listradio]]></question_theme_name>
        <modulename/>
        <same_script><![CDATA[0]]></same_script>
        <language><![CDATA[{lang}]]></language>
      </row>"""

        for score in range(1, 10):
            for lang in ("en", "zh-Hans"):
                answers_rows += f"""
      <row>
        <qid><![CDATA[{bcs2_qid}]]></qid>
        <code><![CDATA[{score}]]></code>
        <answer><![CDATA[{score}]]></answer>
        <sortorder><![CDATA[{score}]]></sortorder>
        <assessment_value><![CDATA[{score}]]></assessment_value>
        <language><![CDATA[{lang}]]></language>
        <scale_id><![CDATA[0]]></scale_id>
      </row>"""
        qid += 1

    # ── 组装完整 LSS ──
    lss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <LimeSurveyDocType>Survey</LimeSurveyDocType>
  <DBVersion>600</DBVersion>
  <languages>
    <language>en</language>
    <language>zh-Hans</language>
  </languages>
  <surveys>
    <fields>
      <fieldname>sid</fieldname>
      <fieldname>gsid</fieldname>
      <fieldname>active</fieldname>
      <fieldname>anonymized</fieldname>
      <fieldname>format</fieldname>
      <fieldname>language</fieldname>
      <fieldname>additional_languages</fieldname>
      <fieldname>datestamp</fieldname>
      <fieldname>showprogress</fieldname>
      <fieldname>showgroupinfo</fieldname>
      <fieldname>showqnumcode</fieldname>
      <fieldname>listpublic</fieldname>
      <fieldname>showwelcome</fieldname>
    </fields>
    <rows>
      <row>
        <sid><![CDATA[{sid}]]></sid>
        <gsid><![CDATA[1]]></gsid>
        <active><![CDATA[N]]></active>
        <anonymized><![CDATA[Y]]></anonymized>
        <format><![CDATA[G]]></format>
        <language><![CDATA[en]]></language>
        <additional_languages><![CDATA[zh-Hans]]></additional_languages>
        <datestamp><![CDATA[Y]]></datestamp>
        <showprogress><![CDATA[Y]]></showprogress>
        <showgroupinfo><![CDATA[B]]></showgroupinfo>
        <showqnumcode><![CDATA[N]]></showqnumcode>
        <listpublic><![CDATA[Y]]></listpublic>
        <showwelcome><![CDATA[Y]]></showwelcome>
      </row>
    </rows>
  </surveys>
  <surveys_languagesettings>
    <fields>
      <fieldname>surveyls_survey_id</fieldname>
      <fieldname>surveyls_language</fieldname>
      <fieldname>surveyls_title</fieldname>
      <fieldname>surveyls_description</fieldname>
      <fieldname>surveyls_welcometext</fieldname>
      <fieldname>surveyls_endtext</fieldname>
    </fields>
    <rows>
      <row>
        <surveyls_survey_id><![CDATA[{sid}]]></surveyls_survey_id>
        <surveyls_language><![CDATA[en]]></surveyls_language>
        <surveyls_title><![CDATA[{SURVEY_TITLE_EN}]]></surveyls_title>
        <surveyls_description><![CDATA[{SURVEY_DESC_EN}]]></surveyls_description>
        <surveyls_welcometext><![CDATA[{SURVEY_WELCOME_EN}]]></surveyls_welcometext>
        <surveyls_endtext><![CDATA[{SURVEY_END_EN}]]></surveyls_endtext>
      </row>
      <row>
        <surveyls_survey_id><![CDATA[{sid}]]></surveyls_survey_id>
        <surveyls_language><![CDATA[zh-Hans]]></surveyls_language>
        <surveyls_title><![CDATA[{SURVEY_TITLE_ZH}]]></surveyls_title>
        <surveyls_description><![CDATA[{SURVEY_DESC_ZH}]]></surveyls_description>
        <surveyls_welcometext><![CDATA[{SURVEY_WELCOME_ZH}]]></surveyls_welcometext>
        <surveyls_endtext><![CDATA[{SURVEY_END_ZH}]]></surveyls_endtext>
      </row>
    </rows>
  </surveys_languagesettings>
  <groups>
    <fields>
      <fieldname>gid</fieldname>
      <fieldname>sid</fieldname>
      <fieldname>group_name</fieldname>
      <fieldname>group_order</fieldname>
      <fieldname>description</fieldname>
      <fieldname>language</fieldname>
      <fieldname>randomization_group</fieldname>
      <fieldname>grelevance</fieldname>
    </fields>
    <rows>{groups_rows}
    </rows>
  </groups>
  <questions>
    <fields>
      <fieldname>qid</fieldname>
      <fieldname>parent_qid</fieldname>
      <fieldname>sid</fieldname>
      <fieldname>gid</fieldname>
      <fieldname>type</fieldname>
      <fieldname>title</fieldname>
      <fieldname>question</fieldname>
      <fieldname>help</fieldname>
      <fieldname>preg</fieldname>
      <fieldname>other</fieldname>
      <fieldname>mandatory</fieldname>
      <fieldname>encrypted</fieldname>
      <fieldname>question_order</fieldname>
      <fieldname>scale_id</fieldname>
      <fieldname>same_default</fieldname>
      <fieldname>relevance</fieldname>
      <fieldname>question_theme_name</fieldname>
      <fieldname>modulename</fieldname>
      <fieldname>same_script</fieldname>
      <fieldname>language</fieldname>
    </fields>
    <rows>{questions_rows}
    </rows>
  </questions>
  <answers>
    <fields>
      <fieldname>qid</fieldname>
      <fieldname>code</fieldname>
      <fieldname>answer</fieldname>
      <fieldname>sortorder</fieldname>
      <fieldname>assessment_value</fieldname>
      <fieldname>language</fieldname>
      <fieldname>scale_id</fieldname>
    </fields>
    <rows>{answers_rows}
    </rows>
  </answers>
</document>"""

    return lss_xml, sid


# ── 主流程 ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LimeSurvey BCS 评分问卷设置（双语 + 条件逻辑）")
    print("=" * 60)

    records = load_dataset()
    print(f"已加载 {len(records)} 条记录")

    print("\n正在构建 LSS XML ...")
    lss_xml, expected_sid = build_lss_xml(records)
    print(f"LSS XML 大小: {len(lss_xml) / 1024:.0f} KB")

    lss_path = os.path.join(BASE_DIR, "survey_structure.lss")
    with open(lss_path, "w", encoding="utf-8") as f:
        f.write(lss_xml)
    print(f"已保存: {lss_path}")

    api = LimeSurveyAPI(LS_URL, LS_USER, LS_PASS)

    try:
        api.connect()

        # 删除已有问卷
        surveys = api.list_surveys()
        if isinstance(surveys, list):
            for s in surveys:
                print(f"  删除已有问卷: {s['sid']}")
                api.delete_survey(int(s["sid"]))

        # 导入
        print("\n正在导入问卷 ...")
        lss_b64 = base64.b64encode(lss_xml.encode("utf-8")).decode("utf-8")
        survey_id = api.import_survey(lss_b64, "lss")

        if isinstance(survey_id, dict):
            print(f"导入失败: {survey_id}")
            return

        print(f"问卷已导入! Survey ID: {survey_id}")

        # 公开
        api.set_survey_properties(survey_id, {"listpublic": "Y"})

        # 验证
        print("\n验证问卷结构 ...")
        groups = api._call("list_groups", [api.session_key, survey_id])
        print(f"  问题组: {len(groups) if isinstance(groups, list) else groups}")

        questions = api._call("list_questions", [api.session_key, survey_id])
        q_count = len(questions) if isinstance(questions, list) else questions
        print(f"  问题数: {q_count}")

        # 检查第一个 BCS 问题
        if isinstance(questions, list) and len(questions) > 0:
            q0 = questions[0]
            props = api.get_question_properties(q0["qid"],
                                                ["title", "question", "answeroptions"])
            title = props.get("title", "?")
            ans = props.get("answeroptions", {})
            has_img = "img" in props.get("question", "") and "src=" in props.get("question", "")
            n_ans = len(ans) if isinstance(ans, dict) and ans != "No available answer options" else 0
            print(f"  {title}: 图片={'✓' if has_img else '✗'}, "
                  f"答案选项={n_ans}{'✓' if n_ans == 9 else '✗'}")

        # 检查条件逻辑（第三个问题）
        if isinstance(questions, list) and len(questions) >= 3:
            q2 = questions[2]
            props2 = api.get_question_properties(q2["qid"],
                                                 ["title", "relevance"])
            rel = props2.get("relevance", "")
            print(f"  {props2.get('title','?')}: relevance={rel}")
            if "NAOK" in rel:
                print(f"  条件逻辑 ✓")

        # 激活
        print("\n正在激活问卷 ...")
        result = api.activate_survey(survey_id)
        print(f"激活结果: {result}")

        print(f"\n{'=' * 60}")
        print(f"问卷创建成功!")
        print(f"Survey ID: {survey_id}")
        print(f"英文: https://shawarmai.com/bcs/index.php/{survey_id}?lang=en")
        print(f"中文: https://shawarmai.com/bcs/index.php/{survey_id}?lang=zh-Hans")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        api.disconnect()


if __name__ == "__main__":
    main()
