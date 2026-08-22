import random
import re

# ===================== 规则库（保留原规则+扩展场景） =====================
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    # 工作场景
    r'I work (as|at) (.*)': [
        "What do you like most about working {0} {1}?",
        "How long have you worked {0} {1}?",
        "Does working {0} {1} bring you a sense of fulfillment?"
    ],
    # 学习场景
    r'I study (.*)': [
        "Why did you choose to study {0}?",
        "What do you find most interesting about {0}?",
        "How do you feel when studying {0}?"
    ],
    # 爱好场景
    r'My hobby is (.*)': [
        "How long have you been interested in {0}?",
        "What do you enjoy most about {0}?",
        "How does {0} make you feel in daily life?"
    ],
    # 情绪场景
    r'I feel (happy|sad|angry|lonely|excited) about (.*)': [
        "Why do you feel {0} about {1}?",
        "What made you start feeling {0} about {1}?",
        "How do you want to change how you feel {0} about {1}?"
    ],
    # 目标场景
    r'I want to (.*) in (.*)': [
        "What steps are you taking to {0} in {1}?",
        "Why is it important for you to {0} in {1}?",
        "How would your life change if you {0} in {1}?"
    ],
    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# ===================== 代词转换 =====================
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}


def swap_pronouns(phrase):
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)


# ===================== 修复后的上下文记忆模块 =====================
user_context = {}

# 非姓名关键词黑名单，避免把职业/状态误判成名字
NOT_NAME_WORDS = {'happy', 'sad', 'angry', 'tired', 'old', 'young',
                  'a', 'an', 'the', 'student', 'teacher', 'doctor',
                  'worker', 'programmer', 'designer', 'engineer'}


def update_context(user_input):
    input_lower = user_input.lower().strip()

    # ---------- 1. 提取姓名（修复：支持 I'm 缩写，优化过滤） ----------
    name_patterns = [
        r'my name is ([a-zA-Z\s\-]{1,20})',  # My name is Tom
        r'i\'m ([a-zA-Z\s\-]{1,20})',  # I'm Tom（新增缩写支持）
        r'i am ([a-zA-Z\s\-]{1,20})'  # I am Tom
    ]

    for pattern in name_patterns:
        match = re.search(pattern, input_lower, re.IGNORECASE)
        if match:
            name_candidate = match.group(1).strip(' .,!?')
            # 拆分单词，只要包含黑名单词汇就不认为是姓名
            name_words = set(name_candidate.lower().split())
            if not name_words & NOT_NAME_WORDS:  # 无交集才是有效姓名
                user_context['name'] = name_candidate
            break  # 匹配到一个就停止

    # ---------- 2. 提取年龄 ----------
    age_match = re.search(r'(\d{1,2}) years? old', input_lower, re.IGNORECASE)
    if age_match:
        user_context['age'] = age_match.group(1)

    # ---------- 3. 提取职业 ----------
    job_match = re.search(r'work (as|at) ([a-zA-Z\s]{1,30})', input_lower, re.IGNORECASE)
    if job_match:
        user_context['job_prep'] = job_match.group(1)
        user_context['job'] = job_match.group(2).strip(' .,!?')


def get_personalized_prefix():
    if not user_context:
        return ""

    prefixes = []
    if 'name' in user_context:
        name = user_context['name']
        prefixes.extend([
            f"By the way {name}, ",
            f"{name}, I was thinking about what you said earlier: ",
            f"Speaking of that {name}, "
        ])
    if 'age' in user_context and 'job' in user_context:
        prefixes.append(f"As a {user_context['age']}-year-old {user_context['job']}, ")
    elif 'age' in user_context:
        prefixes.append(f"At {user_context['age']} years old, ")
    elif 'job' in user_context:
        prefixes.append(f"As a {user_context['job']}, ")

    return random.choice(prefixes) if prefixes else ""


# ===================== 修复后的响应函数 =====================
def respond(user_input):
    update_context(user_input)
    input_lower = user_input.lower()

    # ---------- 修复：记忆查询触发逻辑，支持 what's 缩写 ----------
    is_query_memory = ('what is my' in input_lower
                       or "what's my" in input_lower
                       or 'do you remember' in input_lower)

    if is_query_memory:
        if 'name' in input_lower and 'name' in user_context:
            return f"Your name is {user_context['name']} – I remember that!"
        elif 'age' in input_lower and 'age' in user_context:
            return f"You told me you are {user_context['age']} years old."
        elif ('job' in input_lower or 'work' in input_lower) and 'job' in user_context:
            prep = user_context.get('job_prep', 'as')
            return f"You work {prep} {user_context['job']}, right?"
        else:
            return "I haven't learned that about you yet – tell me more!"

    # ---------- 规则匹配生成基础响应 ----------
    base_response = ""
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            captured_groups = match.groups()
            swapped_groups = [swap_pronouns(group) for group in captured_groups] if captured_groups else []
            if swapped_groups:
                base_response = random.choice(responses).format(*swapped_groups)
            else:
                base_response = random.choice(responses)
            break

    if not base_response:
        base_response = random.choice(rules[r'.*'])

    # ---------- 随机添加个性化前缀 ----------
    if random.random() < 0.3:
        personalized_prefix = get_personalized_prefix()
        if personalized_prefix:
            base_response = personalized_prefix + base_response.lower()

    return base_response


# ===================== 主循环 =====================
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today? (Type 'bye'/'exit'/'quit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Therapist: Goodbye! It was nice talking to you.")
            break
        if not user_input.strip():
            print("Therapist: I'm listening – please share something with me.")
            continue
        print(f"Therapist: {respond(user_input)}")