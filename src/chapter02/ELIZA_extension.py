# ELIZA_extension.py

import random
import re

# ===================== 1. 扩展规则库：新增工作/学习/爱好等场景 =====================
# 定义规则库：模式(正则表达式) -> 响应模板列表
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
    # work
    r'I work (as|at) (.*)': [
        "What do you like most about working {0} {1}?",
        "How long have you worked {0} {1}?",
        "Does working {0} {1} bring you a sense of fulfillment?"
    ],
    # study
    r'I study (.*)': [
        "Why did you choose to study {0}?",
        "What do you find most interesting about {0}?",
        "How do you feel when studying {0}?"
    ],
    # hobby
    r'My hobby is (.*)': [
        "How long have you been interested in {0}?",
        "What do you enjoy most about {0}?",
        "How does {0} make you feel in daily life?"
    ],
    # emotion
    r'I feel (happy|sad|angry|lonely) about (.*)': [
        "Why do you feel {0} about {1}?",
        "What made you start feeling {0} about {1}?",
        "How do you want to change how you feel {0} about {1}?"
    ],
    # goal
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

# ===================== 2. 代词转换规则 =====================
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

# ===================== 3. 上下文记忆功能实现 =====================
# 全局字典存储用户关键信息：姓名/年龄/职业
user_context = {}

def update_context(user_input):
    """从用户输入中提取并更新关键信息(name/age/job)"""
    input_lower = user_input.lower().strip()

    # extract name (match "my name is X" / "I am X" )
    name_patterns = [
        r'my name is ([a-zA-Z\s]{1,32})',
        r'i am ([a-zA-Z\s]{1,32})'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, input_lower, re.IGNORECASE)
        if match:
            name = match.group(1).strip(' .,!?')
            # name filter
            if not any(word in name for word in ['happy', 'sad', 'angry', 'tired', 'young', 'old']):
                user_context['name'] = name
                break

    # extract age (match "X years old" / "I'm X years old" )
    age_match = re.search(r'(\d{1,2}) years? old', input_lower, re.IGNORECASE)
    if age_match:
        user_context['age'] = age_match.group(1)


    # extract job (match "work as X" / "work at X")
    job_match = re.search(r'work (as|at) ([a-zA-Z\s]{1,32})', input_lower, re.IGNORECASE)
    if job_match:
        user_context['job'] = job_match.group(2).strip(' .,!?')

def get_personalized_prefix():
    """随机生成个性化前缀(30%概率触发, 引用记忆的用户信息)"""
    if not user_context:
        return ""

    prefixes = []
    # name prefix
    if 'name' in user_context:
        name = user_context['name']
        prefixes.extend([
            f"By the way {name}, ",
            f"{name}, I was thinking about what you said eariler: ",
            f"Speaking of that {name}, "
        ])
    # age + job mode prefix
    if 'age' in user_context and 'job' in user_context:
        prefixes.append(f"As a {user_context['age']}-year-old {user_context['job']}, ")
    elif 'age' in user_context:
        prefixes.append(f"At {user_context['age']} years old, ")
    elif 'job' in user_context:
        prefixes.append(f"As a {user_context['job']}, ")

    return  random.choice(prefixes) if prefixes else ""

# ===================== 4. 扩展后的响应生成函数 =====================
def respond(user_input):
    """整合上下文记忆+规则匹配的响应生成逻辑"""
    # 1. 更新上下文(提取用户关键信息)
    update_context(user_input)

    # 2. 处理用户主动询问记忆的场景(如"你记得我的名字吗?")
    input_lower = user_input.lower()
    if "what is my" in input_lower or 'do you remember' in input_lower:
        if 'name' in input_lower and 'name' in user_context:
            return f"Your name is {user_context['name']} - I remember that!"
        elif 'age' in input_lower and 'age' in user_context:
            return f"You told me you are {user_context['age']} years old."
        elif ('job' in input_lower or 'work' in input_lower) and 'job' in user_context:
            return f"You work {user_context.get('job_prep', 'as')} a {user_context['job']}, right?"
        else:
            return "I haven't learned that about you yet – tell me more!"

    # 3. 规则匹配生成基础响应
    base_response = ""
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 提取捕获组并处理代词转换
            captured_groups = match.groups()
            swapped_groups = [swap_pronouns(group) for group in captured_groups] if captured_groups else []

            # 格式化响应模板(适配捕获多组)
            if swapped_groups:
                base_response = random.choice(responses).format(*swapped_groups)
            else:
                base_response = random.choice(responses)
            break
    # 兜底:匹配通配符规则
    if not base_response:
        base_response = random.choice(rules[r'.*'])

    # 4. 随机添加个性化前缀(30% 概率)
    if random.random() < 0.3:
        personalized_prefix = get_personalized_prefix()
        if personalized_prefix:
            # 拼接时统一大小写(前缀大写开头,响应小写)
            base_response = personalized_prefix + base_response.lower()

    return base_response

# ===================== 5. 主聊天循环 =====================
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today? (Type 'bye'/'exit'/'quit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Therapist: Goodbye! It was nice talking to you.")
            break
        # 空输入处理
        if not user_input.strip():
            print("Therapist: I'm listening – please share something with me.")
            continue
            # 生成并输出响应
        response = respond(user_input)
        print(f"Therapist: {response}")