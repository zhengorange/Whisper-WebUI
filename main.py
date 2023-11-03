import gradio as gr
from glob import glob
import datetime
import argparse
from chat import TPUChatglm
import subprocess
import re
import os

class App:

    text = ""
    llm = None

    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(title="bmwhisper", theme=self.args.theme)
    
    def start_trans_file(self, fileobj):
        file_name = fileobj[0].name
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        console_result = subprocess.run(['bmwhisper', file_name, "--model", "small", "--output_dir", "outputs/" + current_time, "--bmodel_dir", "/data/whisper-TPU_py/bmodel", "--chip_mode", "soc", "--verbose", "False"], capture_output=True, text=True)
        # print(console_result)
        content = ""
        with open(glob(f"./outputs/{current_time}/*.txt")[0], 'r') as file:
            content = file.read()
        print("time:", current_time)
        print("content:", content)
        self.text = content
        return content, glob(f"./outputs/{current_time}/*"), gr.update(visible=True), gr.update(visible=True)

    def start_trans_file_from_host(self, dir_input, fileobj):
            if os.path.isdir(f"{dir_input}/{fileobj}"):
                gr.Info('Please choose file item instead of folder item.')
                return None, None, None, None
            
            file_name = f"{dir_input}/{fileobj}"
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            console_result = subprocess.run(['bmwhisper', file_name, "--model", "small", "--output_dir", "outputs/" + current_time, "--bmodel_dir", "/data/whisper-TPU_py/bmodel", "--chip_mode", "soc", "--verbose", "False"], capture_output=True, text=True)
            # print(console_result)
            content = ""
            with open(glob(f"./outputs/{current_time}/*.txt")[0], 'r') as file:
                content = file.read()
            print("time:", current_time)
            print("content:", content)
            self.text = content
            return content, glob(f"./outputs/{current_time}/*"), gr.update(visible=True), gr.update(visible=True)
    
    def summary(self):
        if self.llm == None:
            self.llm = TPUChatglm()

        if len(self.text) < 900:
            prompt = f"请使用中文摘要下面这段文字：\n\n{self.text}\n" 
            res = ""
            for response, _ in self.llm.stream_predict(prompt, []):
                res = response
            return res
        else:
            segments = []
            current_segment = ''
            all_segments = [x for x in re.split(r'[。.]', self.text) if x != ""]

            for sentence in all_segments:
                if len(current_segment) + len(sentence) <= 800:
                    current_segment += sentence + '。'
                else:
                    segments.append(current_segment)
                    current_segment = sentence + '。'

            segments.append(current_segment)
            summaries = []
            for item in segments:
                prompt = f"请使用中文摘要下面这段文字：\n\n{item}\n" 
                res = ""
                for response, _ in self.llm.stream_predict(prompt, []):
                    res = response
                summaries.append(res)

            return "\n".join(summaries)

    def summary1(self):
        if self.llm == None:
            self.llm = TPUChatglm()

        if len(self.text) < 900:
            prompt = f"请使用中文摘要下面这段文字：\n\n{self.text}\n" 
            res = ""
            for response, _ in self.llm.stream_predict(prompt, []):
                res = response
            return res
        else:
            segments = []
            current_segment = ''
            all_segments = [x for x in re.split(r'[。.]', self.text) if x != ""]

            for sentence in all_segments:
                if len(current_segment) + len(sentence) <= 800:
                    current_segment += sentence + '。'
                else:
                    segments.append(current_segment)
                    current_segment = sentence + '。'

            segments.append(current_segment)
            summaries = []
            for item in segments:
                prompt = f"请使用中文摘要下面这段文字：\n\n{item}\n" 
                res = ""
                for response, _ in self.llm.stream_predict(prompt, []):
                    res = response
                summaries.append(res)

            return "\n".join(summaries)
            

    def clear(self):
        self.text = ""
        return None, None, None, None, gr.update(visible=False), gr.update(visible=False)

    def clear1(self):
            self.text = ""
            return None, None, None, None, gr.update(visible=False), gr.update(visible=False)
    
    def dir_click(self, dir_inp):
        files = os.listdir(dir_inp)
        return gr.update(choices=files, value=None)
    
    def enter_dir(self, dir_input, input_file):
        if os.path.isdir(f"{dir_input}/{input_file}"):
            if dir_input == "/":
                dir_input = ""
            files = os.listdir(f"{dir_input}/{input_file}")
            return f"{dir_input}/{input_file}", gr.update(choices=files, value=None)
        else:
            gr.Info('Please choose folder item instead of file item.')
            return dir_input, input_file
    

    def launch(self):
        with self.app:
            gr.Markdown("# BMWhisper")
            with gr.Row():
                with gr.Column():
                    gr.Markdown(elem_id="md_project")
            with gr.Tabs():
                with gr.TabItem("Select File From Browser"):
                    with gr.Row():
                        input_file = gr.Files(type="filepath", label="Upload File here", file_types=["audio"])                
                    with gr.Row():
                        btn_run = gr.Button("开始转录", variant="primary")
                        clear_run = gr.Button("清除")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="转录文本", scale=8)
                    with gr.Row():
                        summary_run = gr.Button("摘要文本", visible=False)
                    with gr.Row():
                        summary_indicator = gr.Textbox(label="文本摘要", scale=8, visible=False)
                    with gr.Row():
                        download = gr.Files(label="已转录文档")

                    btn_run.click(fn=self.start_trans_file, inputs=[input_file], outputs=[tb_indicator, download, summary_run, summary_indicator])
                    clear_run.click(fn=self.clear, outputs=[tb_indicator, summary_indicator, input_file, download, summary_run, summary_indicator])
                    summary_run.click(fn=self.summary, outputs=[summary_indicator])
                
                with gr.TabItem("Select File From Serve Host"):
                    with gr.Row():         
                        with gr.Column(scale=4):
                            dir_input = gr.Textbox(label="Check A Folder On The Host")
                            dir_btn = gr.Button(value="Check")
                        with gr.Column(scale=4):
                            input_file = gr.Dropdown(label="File List", choices=[], value=None, interactive=True)
                            enter_dir = gr.Button(value="Enter Folder")

                    with gr.Row():
                        btn_run = gr.Button("开始转录", variant="primary")
                        clear_run = gr.Button("清除")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="转录文本", scale=8)
                    with gr.Row():
                        summary_run = gr.Button("摘要文本", visible=False)
                    with gr.Row():
                        summary_indicator = gr.Textbox(label="文本摘要", scale=8, visible=False)
                    with gr.Row():
                        download = gr.Files(label="已转录文档")
                    
                    dir_btn.click(fn=self.dir_click, inputs=[dir_input], outputs=[input_file])
                    enter_dir.click(fn=self.enter_dir, inputs=[dir_input, input_file], outputs=[dir_input, input_file])
                    btn_run.click(fn=self.start_trans_file_from_host, inputs=[dir_input, input_file], outputs=[tb_indicator, download, summary_run, summary_indicator])
                    clear_run.click(fn=self.clear1, outputs=[tb_indicator, summary_indicator, input_file, download, summary_run, summary_indicator])
                    summary_run.click(fn=self.summary1, outputs=[summary_indicator])


        # Launch the app with optional gradio settings
        launch_args = {}
        if self.args.share:
            launch_args['share'] = self.args.share
        if self.args.server_name:
            launch_args['server_name'] = self.args.server_name
        if self.args.server_port:
            launch_args['server_port'] = self.args.server_port
        if self.args.username and self.args.password:
            launch_args['auth'] = (self.args.username, self.args.password)
        self.app.queue(api_open=False).launch(**launch_args)


# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default="0.0.0.0", help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
