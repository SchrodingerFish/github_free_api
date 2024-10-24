### 1. request header should be Authorization: Bearer <token>,and the <token> should be like this: project_id#api_key,(tips: you can get the project_id and api_key from customgpt.ai)
### 2. only support models:gpt-4-o,
### 3. only support stream or no-stream chat,
### 4. run command uvicorn main:app --host 0.0.0.0 --port 8080 --reload   server will run at port 8080 then just enjoy it!
### 5. you can build image with the Dockerfile too.
