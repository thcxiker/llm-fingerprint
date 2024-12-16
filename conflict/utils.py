def extract_after_word(s, word):
    # 找到单词在字符串中的起始位置
    index = s.find(word)
    
    # 如果单词存在，返回从该单词之后的部分
    if index != -1:
        return s[index + len(word):].strip()  # `strip()` 去除前后多余空格
    else:
        return None  # 如果单词不存在，返回 None 或空字符串
import re

def extract_option_letter(text):
    """
    提取文本中的选项字母A 到 E。
    
    支持以下两种格式：
    1. 带有固定前缀和括号，例如 "The closest option is D) Susan Frampton."
    2. 仅有字母选项，例如 "D"
    
    参数:
        text (str): 输入的字符串。
    
    返回:
        str: 匹配的选项字母A 到 E，如果没有匹配则返回 None。
    """
    match1 = re.search(r"(?:option is )?([A-E])(?:\))?", text)
    match2 = re.search(r"(?:option)?([A-E])(?:\))?", text)

    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    return None

# # 示例调用
# text1 = "The closest option is D) Susan Frampton."
# text2 = "D"
# print(extract_option_letter(text1))  # 输出: D
# print(extract_option_letter(text2))  # 输出: D

