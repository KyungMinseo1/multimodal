from graph import plot_feedback_and_df
from llm_prompt import LLMPipeline
import json
import markdown

llm_model = LLMPipeline()

class GenerateFeedback:
    def __init__(self):
        pass
    
    def generate(self, topic:str, name:str, data:list) -> str:
        """
        topic : str type topic info
        name : str type user name
        data : list type model inference data
        """
        result = sorted(data, key = lambda x : x['timestamp']['start'])
        data_str, stt_chunk = plot_feedback_and_df(result, name)
        feedback = llm_model.print_report(topic, data_str, stt_chunk)
        assert feedback is not None
        markdown_feedback = markdown.markdown(feedback)
        return markdown_feedback
    

if __name__ == "__main__":
    with open("analysis/result_경민서_modified.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    gf = GenerateFeedback()
    markdown_feedback = gf.generate("경영정보시스템", "경민서", data)
    print(markdown_feedback)