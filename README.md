# What's this
This project can build a simple Web application that, by sending a POST request carrying user_id (a random number to avoid different conversation processes using the same conversation record), speaker_id, and user_question, can obtain answers from the ChatGPT model (default: gpt-3.5-turbo), including text and audio generated through VITS inference. 

This project is based on the [MoeGoe](https://github.com/CjangCjengh/MoeGoe), and the VITS model used for inference comes from https://github.com/Plachtaa/VITS-fast-fine-tuning/tree/main. Many thanks for this.

# How to use
1.Clone this repository
```
git clone https://github.com/kagari-bi/Vits_Webapp.git
```
2.Download the pre-trained model (https://pan.baidu.com/s/1fl9504KPKlnZFE6Ix8HXjw?pwd=csyc) and unzip it to the models\Umamusume directory.

3.Open config_backup.ini, enter your OpenAI account's api_key, Baidu account's appid and key (used to translate Chatgpt's response into Japanese, then use vits for inference), and proxy address, etc. Save and close, then rename it to config.ini.

4.Install requirements
```
pip install -r requirements.txt
```
5.Run the Web application
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# A Simple Example
After running the Web application, you can try using PostRequest.ipynb to understand the format of Post forms and responses.
