import aiohttp
import asyncio
import os

API_URL = "http://localhost:8765/api/image_ocr"
FILE_DIR = "./data/input/"
DOCKER_DIR = "./data/input/"

async def send_one(session, item):
    data = {"files": [item]}
    print(f"\n发送文件 {item['id']} {item['file_path']} ...")

    try:
        async with session.post(API_URL, json=data, timeout=600000000) as resp:
            text = await resp.text()
            print("响应状态码:", resp.status)
            print("响应内容:", text)
    except Exception as e:
        print(f"请求失败: {e}")

async def main():
    files = []
    for i, file in enumerate(os.listdir(FILE_DIR)):
        file_path = os.path.join(DOCKER_DIR, file)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            file_id = str(i)
            item = {"id": file_id, "file_path": file_path}
            files.append(item)
            print(f"准备文件: {file_path}")

    print(f"总文件数：{len(files)}")

    async with aiohttp.ClientSession() as session:
        tasks = [send_one(session, item) for item in files]
        # 并发运行所有任务
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
